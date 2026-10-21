/*
 * Copyright (c) 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#ifndef _TORX_DEVICE_H_
#define _TORX_DEVICE_H_

#include <stdint.h>

#include "util/macros.h"

PUBLIC
const struct pipe_ml_device *torx_device_probe(void);

PUBLIC
const uint32_t *torx_abi_layout(unsigned *count);

#endif /* _TORX_DEVICE_H_ */