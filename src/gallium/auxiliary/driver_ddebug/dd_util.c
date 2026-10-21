/**************************************************************************
 *
 * Copyright 2015 Advanced Micro Devices, Inc.
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include "dd_util.h"

#include "pipe/p_screen.h"
#include "util/os_file.h"
#include "util/u_atomic.h"
#include "util/u_process.h"
#include "c99_alloca.h"
#include "util/detect_os.h"

#if DETECT_OS_WINDOWS
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void
dd_get_debug_filename_and_mkdir(char *buf, size_t buflen, bool verbose)
{
   static unsigned index;
   char dir[256];
   const char *proc_name = util_get_process_name();

   if (!proc_name) {
      fprintf(stderr, "dd: can't get the process name\n");
      proc_name = "unknown";
   }

   snprintf(dir, sizeof(dir), "%s/"DD_DIR, debug_get_option("HOME", "."));

   if (os_mkdir(dir, 0774) && errno != EEXIST)
      fprintf(stderr, "dd: can't create a directory (%i)\n", errno);

   snprintf(buf, buflen, "%s/%s_%u_%08u", dir, proc_name, (unsigned int)getpid(),
            (unsigned int)p_atomic_inc_return(&index) - 1);

   if (verbose)
      fprintf(stderr, "dd: dumping to file %s\n", buf);
}

FILE *
dd_get_debug_file(bool verbose)
{
   char name[512];
   FILE *f;

   dd_get_debug_filename_and_mkdir(name, sizeof(name), verbose);
   f = fopen(name, "w");
   if (!f) {
      fprintf(stderr, "dd: can't open file %s\n", name);
      return NULL;
   }

   return f;
}

void
dd_parse_apitrace_marker(const char *string, int len, unsigned *call_number)
{
   unsigned num;
   char *s;

   if (len <= 0)
      return;

   /* Make it zero-terminated. */
   s = alloca(len + 1);
   memcpy(s, string, len);
   s[len] = 0;

   /* Parse the number. */
   errno = 0;
   num = strtol(s, NULL, 10);
   if (errno)
      return;

   *call_number = num;
}

void
dd_write_header(FILE *f, struct pipe_screen *screen, unsigned apitrace_call_number)
{
   char cmd_line[4096];
   if (util_get_command_line(cmd_line, sizeof(cmd_line)))
      fprintf(f, "Command: %s\n", cmd_line);
   fprintf(f, "Driver vendor: %s\n", screen->get_vendor(screen));
   fprintf(f, "Device vendor: %s\n", screen->get_device_vendor(screen));
   fprintf(f, "Device name: %s\n\n", screen->get_name(screen));

   if (apitrace_call_number)
      fprintf(f, "Last apitrace call: %u\n\n", apitrace_call_number);
}
