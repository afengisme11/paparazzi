/*
 * Copyright (C) Shuo Li
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
 * @file "modules/state_autonomous_race/state_autonomous_race.c"
 * @author Shuo Li
 * The module is used to store all states in the competition
 */

#include "modules/state_autonomous_race/state_autonomous_race.h"
#include "firmwares/rotorcraft/autopilot.h"
#include "modules/stereocam/stereo_gate_position/stereo_gate_position.h"
#include "modules/command_level_iros/command_level_iros.h"
#include <stdio.h>
#include "modules/command_level_iros/command_level_iros.h"

void state_autonomous_race_init();
void display_lower_state();



struct state_autonomous_race states_race;

void state_autonomous_race_init() {
    states_race.gate_counter = 0;
    states_race.ready_pass_through = 0;
    states_race.turning = 0;
}

void display_states()
{
    if (autopilot_mode != AP_MODE_MODULE)
        return;
   // printf("gate_counter is %d \n",states_race.gate_counter);
    display_lower_state();
    printf("\n");
    printf("\n");
    printf("\n");

}

void display_lower_state()
{
    switch(state_lower_level){
        case WAIT_FOR_DETECTION_CM:
            printf("It is in WAIT_FOR_DETECTION\n");
            break;
        case ADJUST_POSITION_CM:
            printf("It is in ADJUST_POSITION\n");
            break;
        case GO_THROUGH_CM:
            printf("It is in GO_THROUGH\n");
            break;
        case HOVER_CM:
            printf("It is in HOVER\n");
            break;
        case TURN_CM:
            printf("It is in TURN\n");
            break;
        case SEARCH_GATE_CM:
            printf("It is in SEARCH_GATE\n");
            break;
    }
}