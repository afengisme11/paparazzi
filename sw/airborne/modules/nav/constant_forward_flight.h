/*
 * Copyright (C) C. DW
 *
 * This file is part of paparazzi
 *
 * paparazzi is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * paparazzi is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with paparazzi; see the file COPYING.  If not, see
 * <http://www.gnu.org/licenses/>.
 */
/**
 * @file "modules/nav/constant_forward_flight.h"
 * @author C. DW
 * 
 */

#ifndef CONSTANT_FORWARD_FLIGHT_H
#define CONSTANT_FORWARD_FLIGHT_H

#include "std.h"

bool_t mod_avoid_init(uint8_t _wp);
bool_t mod_avoid_run(void);

extern void forward_flight_init(void);
extern void forward_flight_periodic(void);

#endif

