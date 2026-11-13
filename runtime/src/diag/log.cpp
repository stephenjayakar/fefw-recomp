// SPDX-License-Identifier: GPL-3.0-or-later
#include "nxrt/log.h"

#include "nxrt/config.h"

#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace nxrt {
namespace {

constexpr size_t kMaxMessage = 2000;     // longer messages are cut to this many bytes and get "..."
constexpr size_t kFormatBuffer = 4096;   // vsnprintf target; anything longer is cut anyway
constexpr unsigned kFlushEvery = 100;    // lines between flushes for Info and below
constexpr char kLevelLetters[] = "EWIDT";
constexpr int kLevelCount = 5;

// One lock (g_mu) guards the file, counters, sink and time base. g_level and g_inited are atomics so the
// level check on the hot path takes no lock. Config is read only before g_mu is taken (see resolve()).
std::recursive_mutex g_mu;   // recursive so a sink that logs does not deadlock
std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};
std::atomic<bool> g_inited{false};
std::FILE* g_file = nullptr;
std::string g_path;          // path of the open file; empty when none
LogSink g_sink = nullptr;
void* g_sink_user = nullptr;
bool g_to_stderr = true;
bool g_stderr_all = false;
uint64_t g_counts[kLevelCount] = {};
unsigned g_since_flush = 0;
long long g_freq = 1;
long long g_t0 = 0;         // QueryPerformanceCounter at log_init

thread_local char t_name[64] = {};

struct Resolved {
    std::string path;
    LogLevel level;
    bool to_stderr;
    bool stderr_all;
    bool keep_previous;
};

// Final settings: env > [log] config keys > options. Reads config, so it must be called without g_mu held.
Resolved resolve(const LogOptions& opt) {
    Resolved r{};
    r.path = opt.path ? opt.path : "nxrt.log";
    r.level = opt.level;
    r.to_stderr = opt.to_stderr;
    r.stderr_all = opt.stderr_all;
    r.keep_previous = opt.keep_previous;
    if (const char* e = std::getenv("NXRT_LOG")) {
        r.path = e;
    } else if (config_has("log", "path")) {
        r.path = config_get_str("log", "path");
    }
    if (const char* e = std::getenv("NXRT_LOG_LEVEL")) {
        r.level = log_parse_level(e, r.level);
    } else if (config_has("log", "level")) {
        std::string lv = config_get_str("log", "level");
        r.level = log_parse_level(lv.c_str(), r.level);
    }
    return r;
}

// Moves an existing file to <path>.prev, replacing any older .prev.
void rotate_previous(const std::string& path) {
    std::FILE* old = std::fopen(path.c_str(), "rb");
    if (!old) return;
    std::fclose(old);
    std::string prev = path + ".prev";
    std::remove(prev.c_str());
    std::rename(path.c_str(), prev.c_str());
}

// Applies resolved settings: closes the old file, resets counters, opens the new file.
void apply(const Resolved& r, bool only_if_uninit) {
    std::lock_guard<std::recursive_mutex> g(g_mu);
    if (only_if_uninit && g_inited.load(std::memory_order_acquire)) return;
    if (g_file) {
        std::fclose(g_file);
        g_file = nullptr;
    }
    g_path.clear();
    g_level.store(static_cast<int>(r.level), std::memory_order_relaxed);
    g_to_stderr = r.to_stderr;
    g_stderr_all = r.stderr_all;
    std::memset(g_counts, 0, sizeof g_counts);
    g_since_flush = 0;

    LARGE_INTEGER freq{}, now{};
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    g_freq = freq.QuadPart > 0 ? freq.QuadPart : 1;
    g_t0 = now.QuadPart;

    if (!r.path.empty()) {
        if (r.keep_previous) rotate_previous(r.path);
        if (std::FILE* f = std::fopen(r.path.c_str(), "w")) {
            g_file = f;
            g_path = r.path;
        } else {
            std::fprintf(stderr, "nxrt: cannot open log file '%s'\n", r.path.c_str());
        }
    }
    g_inited.store(true, std::memory_order_release);
}

double seconds_since_init_locked() {
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart - g_t0) / static_cast<double>(g_freq);
}

}  // namespace

void log_init(const LogOptions& opt) {
    Resolved r = resolve(opt);  // config reads happen before the log lock is taken
    apply(r, false);
}

void log_shutdown() {
    std::lock_guard<std::recursive_mutex> g(g_mu);
    if (g_file) {
        std::fflush(g_file);
        std::fclose(g_file);
        g_file = nullptr;
    }
    g_path.clear();
    // g_inited stays set: later logf calls do not re-open anything until log_init is called again.
}

void log_flush() {
    std::lock_guard<std::recursive_mutex> g(g_mu);
    if (g_file) std::fflush(g_file);
}

void log_set_level(LogLevel l) { g_level.store(static_cast<int>(l), std::memory_order_relaxed); }

LogLevel log_level() { return static_cast<LogLevel>(g_level.load(std::memory_order_relaxed)); }

bool log_enabled(LogLevel l) { return static_cast<int>(l) <= g_level.load(std::memory_order_relaxed); }

const char* log_path() {
    std::lock_guard<std::recursive_mutex> g(g_mu);
    return g_path.c_str();  // valid until the next log_init or log_shutdown
}

void log_set_thread_name(const char* name) {
    std::snprintf(t_name, sizeof t_name, "%s", name ? name : "");
    // Also the Windows thread description, so the [diag] budget thread table and external profilers see the runtime's worker names.
    using SetDescFn = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const SetDescFn set_desc = reinterpret_cast<SetDescFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription")));
    if (set_desc && t_name[0]) {
        wchar_t w[sizeof t_name];
        size_t i = 0;
        for (; t_name[i] && i + 1 < sizeof t_name; ++i) w[i] = static_cast<wchar_t>(static_cast<unsigned char>(t_name[i]));
        w[i] = 0;
        set_desc(GetCurrentThread(), w);
    }
}

const char* log_thread_name() { return t_name; }

static std::string trimmed_lower(const char* s) {
    std::string v = s;
    size_t b = v.find_first_not_of(" \t\r\n");
    size_t e = v.find_last_not_of(" \t\r\n");
    v = (b == std::string::npos) ? std::string() : v.substr(b, e - b + 1);
    for (char& c : v)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return v;
}

LogLevel log_parse_level(const char* s, LogLevel fallback) {
    if (!s) return fallback;
    std::string v = trimmed_lower(s);
    if (v == "error") return LogLevel::Error;
    if (v == "warn") return LogLevel::Warn;
    if (v == "info") return LogLevel::Info;
    if (v == "debug") return LogLevel::Debug;
    if (v == "trace") return LogLevel::Trace;
    if (v.size() == 1 && v[0] >= '0' && v[0] <= '4') return static_cast<LogLevel>(v[0] - '0');
    return fallback;
}

const char* log_level_name(LogLevel l) {
    switch (l) {
        case LogLevel::Error: return "error";
        case LogLevel::Warn: return "warn";
        case LogLevel::Info: return "info";
        case LogLevel::Debug: return "debug";
        case LogLevel::Trace: return "trace";
    }
    return "?";
}

uint64_t log_count(LogLevel l) {
    int i = static_cast<int>(l);
    if (i < 0 || i >= kLevelCount) return 0;
    std::lock_guard<std::recursive_mutex> g(g_mu);
    return g_counts[i];
}

void log_set_sink(LogSink sink, void* user) {
    std::lock_guard<std::recursive_mutex> g(g_mu);
    g_sink = sink;
    g_sink_user = user;
}

void logv(LogLevel l, const char* tag, const char* fmt, va_list ap) {
    if (!g_inited.load(std::memory_order_acquire)) {
        Resolved r = resolve(LogOptions{});  // first use: defaults, plus env and config
        apply(r, true);
    }
    const int li = static_cast<int>(l);
    if (li < 0 || li >= kLevelCount) return;
    if (li > g_level.load(std::memory_order_relaxed)) return;  // filtered: not counted, not written

    char body[kFormatBuffer];
    body[0] = '\0';
    std::vsnprintf(body, sizeof body, fmt ? fmt : "", ap);
    std::string msg = body;
    for (char& c : msg)
        if (c == '\n' || c == '\r') c = ' ';
    if (msg.size() > kMaxMessage) {
        size_t cut = kMaxMessage;
        while (cut > 0 && (static_cast<unsigned char>(msg[cut]) & 0xC0) == 0x80) --cut;  // do not split UTF-8
        msg.resize(cut);
        msg += "...";
    }

    std::lock_guard<std::recursive_mutex> g(g_mu);
    char prefix[256];
    std::snprintf(prefix, sizeof prefix, "[%10.6f %c t%lu %s] ", seconds_since_init_locked(), kLevelLetters[li],
                  static_cast<unsigned long>(GetCurrentThreadId()), t_name[0] ? t_name : "-");
    std::string line = prefix;
    line += tag ? tag : "";
    line += ": ";
    line += msg;

    g_counts[li]++;
    if (g_file) {
        std::fwrite(line.data(), 1, line.size(), g_file);
        std::fputc('\n', g_file);
        if (li <= static_cast<int>(LogLevel::Warn) || ++g_since_flush >= kFlushEvery) {
            std::fflush(g_file);
            g_since_flush = 0;
        }
    }
    if (g_to_stderr && (g_stderr_all || li <= static_cast<int>(LogLevel::Warn))) {
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    if (g_sink) g_sink(l, line.c_str(), g_sink_user);
}

void logf(LogLevel l, const char* tag, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    logv(l, tag, fmt, ap);
    va_end(ap);
}

}  // namespace nxrt
