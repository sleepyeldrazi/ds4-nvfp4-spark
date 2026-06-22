#ifndef DS4_ENV_H
#define DS4_ENV_H

#include <stdlib.h>

/* One-shot cached getenv() helpers.
 *
 * The decode path was paying for ~150 libc getenv() scans per token; these
 * macros memoize the result in a per-use-site static variable so the hot path
 * becomes a single integer read.  The cache is populated on first use and is
 * never refreshed, which matches the semantics of all DS4 diagnostic/override
 * environment flags.
 */
#ifdef __GNUC__
#define DS4_ENV_BOOL(NAME) __extension__({ \
    static int _ds4_env_cached = -1; \
    if (_ds4_env_cached < 0) _ds4_env_cached = getenv(NAME) != NULL; \
    _ds4_env_cached; \
})

#define DS4_ENV_UINT(NAME, DEFAULT) __extension__({ \
    static int _ds4_env_cached = -1; \
    if (_ds4_env_cached < 0) { \
        const char *_e = getenv(NAME); \
        _ds4_env_cached = _e ? (int)strtoul(_e, NULL, 10) : (int)(DEFAULT); \
    } \
    (unsigned int)_ds4_env_cached; \
})

#define DS4_ENV_STR(NAME) __extension__({ \
    static const char *_ds4_env_cached = NULL; \
    static int _ds4_env_init = 0; \
    if (!_ds4_env_init) { _ds4_env_init = 1; _ds4_env_cached = getenv(NAME); } \
    _ds4_env_cached; \
})
#else
#define DS4_ENV_BOOL(NAME)  (getenv(NAME) != NULL)
#define DS4_ENV_UINT(NAME, DEFAULT) \
    ((unsigned int)(getenv(NAME) ? strtoul(getenv(NAME), NULL, 10) : (DEFAULT)))
#define DS4_ENV_STR(NAME)   (getenv(NAME))
#endif

#endif /* DS4_ENV_H */
