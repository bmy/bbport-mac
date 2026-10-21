/* -*- mesa-c++  -*-
 * Copyright 2018 Collabora LTD
 * Author: Gert Wollny <gert.wollny@collabora.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef R600_DUMP_H
#define R600_DUMP_H

#include <stdio.h>

struct r600_shader;

void print_shader_info(FILE *f , int id, struct r600_shader *shader);

#endif // R600_DUMP_H
