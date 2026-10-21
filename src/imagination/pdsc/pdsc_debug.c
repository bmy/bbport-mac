/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pdsc_debug.c
 *
 * \brief PDS compiler debug-related functions.
 */

#include "pdsc_internal.h"

#include "util/u_call_once.h"
#include "util/u_debug.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const struct debug_named_value pdsc_debug_options[] = {
   { "val_skip", PDSC_DEBUG_VAL_SKIP, "Skip IR validation." },
   { "print", PDSC_DEBUG_PRINT, "Print the PDS IR." },
   { "raw_regs", PDSC_DEBUG_RAW_REGS, "Print raw regs, not names." },
   { "print_binary", PDSC_DEBUG_PRINT_BINARY, "Print the PDS binary." },
   DEBUG_NAMED_VALUE_END,
};

DEBUG_GET_ONCE_FLAGS_OPTION(pdsc_debug, "PDSC_DEBUG", pdsc_debug_options, 0U)
uint64_t pdsc_debug = 0U;

DEBUG_GET_ONCE_OPTION(pdsc_color, "PDSC_COLOR", NULL)
bool pdsc_color = false;

static void pdsc_debug_init_once(void)
{
   /* Get debug flags. */
   pdsc_debug = debug_get_option_pdsc_debug();

   /* Get/parse color option. */
   const char *color_opt = debug_get_option_pdsc_color();
   if (!color_opt || !strcmp(color_opt, "auto") || !strcmp(color_opt, "a"))
      pdsc_color = isatty(fileno(stdout));
   else if (!strcmp(color_opt, "on") || !strcmp(color_opt, "1"))
      pdsc_color = true;
   else if (!strcmp(color_opt, "off") || !strcmp(color_opt, "0"))
      pdsc_color = false;
}

void pdsc_debug_init(void)
{
   static util_once_flag flag = UTIL_ONCE_FLAG_INIT;
   util_call_once(&flag, pdsc_debug_init_once);
}
