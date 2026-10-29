// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* Local trophy store: every unlock is appended as one JSON line to <user>/trophies.jsonl.
 * Opt-in companion tools may read that file; the runtime itself never touches the network. */
#ifndef BB_RUNTIME_TROPHY_H
#define BB_RUNTIME_TROPHY_H
#include <stdint.h>

/* Appends one unlock event as a JSON line to <user_dir>/trophies.jsonl.
   Returns 0 on success, -1 on failure. Never aborts. Thread-safe. */
int trophy_store_append(const char *user_dir, const char *title_id, int32_t trophy_id);

/* Remembers the TITLE_ID read from param.sfo; invalid or missing values keep the fallback. */
void trophy_store_configure(const char *title_id);
/* TITLE_ID used for unlock events: the configured one, or TROPHY_STORE_DEFAULT_TITLE. */
const char *trophy_store_title_id(void);

#define TROPHY_STORE_DEFAULT_TITLE "CUSA03173"
#define TROPHY_STORE_FILE "trophies.jsonl"
#endif
