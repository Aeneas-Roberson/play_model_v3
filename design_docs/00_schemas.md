# offense_embedding

| Column | Data Type |
|--------|-----------|
| play_id | INTEGER |
| game_id | INTEGER |
| offense_team_id | INTEGER |
| offense_conference | TEXT |
| home_away_indicator | INTEGER |
| coach_first_name | TEXT |
| coach_last_name | TEXT |
| years_at_school | REAL |
| coach_total_experience | REAL |
| new_coach_indicator | INTEGER |
| talent_zscore | REAL |
| run_rate_1st_down | REAL |
| run_rate_2nd_short | REAL |
| run_rate_2nd_medium | REAL |
| run_rate_2nd_long | REAL |
| run_rate_3rd_short | REAL |
| run_rate_3rd_medium | REAL |
| run_rate_3rd_long | REAL |
| punt_rate_4th_short | REAL |
| punt_rate_4th_medium | REAL |
| punt_rate_4th_long | REAL |
| fg_attempt_rate_by_field_position | REAL |
| go_for_it_rate_4th_down | REAL |
| go_for_2_rate | REAL |
| onside_kick_rate | REAL |
| fake_punt_rate | REAL |
| avg_seconds_per_play | REAL |
| plays_per_game | REAL |
| penalty_rate | REAL |
| penalty_yards_per_game | REAL |
| recent_avg_seconds_per_play | REAL |
| recent_plays_per_game | REAL |
| recent_penalty_rate | REAL |
| recent_run_rate_by_down_distance | REAL |
| opponent_wins | INTEGER |
| opponent_losses | INTEGER |
| home_wins | INTEGER |
| home_losses | INTEGER |
| away_wins | INTEGER |
| away_losses | INTEGER |
| conference_wins | INTEGER |
| conference_losses | INTEGER |
| avg_opponent_talent_rating | REAL |
| avg_opponent_talent_rating_of_wins | REAL |
| avg_opponent_talent_rating_of_losses | REAL |
| strength_of_schedule | REAL |
| wins_vs_favored_opponents | INTEGER |
| losses_vs_weaker_opponents | INTEGER |
| point_differential_vs_talent_expectation | REAL |
| created_at | TIMESTAMP |
| updated_at | TIMESTAMP |
|year| INTEGER|
|week| INTEGER|

# defense_embedding

| Column | Data Type |
|--------|-----------|
| play_id | INTEGER |
| game_id | INTEGER |
| defense_team_id | INTEGER |
| defense_conference | TEXT |
| defense_coach_first_name | TEXT |
| defense_coach_last_name | TEXT |
| defense_years_at_school | REAL |
| defense_coach_total_experience | REAL |
| defense_new_coach_indicator | INTEGER |
| defense_talent_zscore | REAL |
| defense_run_stop_rate_1st_down | REAL |
| defense_run_stop_rate_2nd_short | REAL |
| defense_run_stop_rate_2nd_medium | REAL |
| defense_run_stop_rate_2nd_long | REAL |
| defense_run_stop_rate_3rd_short | REAL |
| defense_run_stop_rate_3rd_medium | REAL |
| defense_run_stop_rate_3rd_long | REAL |
| defense_red_zone_fg_rate | REAL |
| defense_red_zone_stop_rate | REAL |
| defense_avg_seconds_allowed_per_play | REAL |
| defense_plays_allowed_per_game | REAL |
| defense_penalty_rate | REAL |
| defense_penalty_yards_per_game | REAL |
| defense_recent_avg_seconds_allowed_per_play | REAL |
| defense_recent_plays_allowed_per_game | REAL |
| defense_recent_penalty_rate | REAL |
| defense_recent_stop_rate_by_down_distance | REAL |
| defense_opponent_wins | INTEGER |
| defense_opponent_losses | INTEGER |
| defense_home_wins | INTEGER |
| defense_home_losses | INTEGER |
| defense_away_wins | INTEGER |
| defense_away_losses | INTEGER |
| defense_conference_wins | INTEGER |
| defense_conference_losses | INTEGER |
| defense_avg_opponent_talent_rating | REAL |
| defense_avg_opponent_talent_rating_of_wins | REAL |
| defense_avg_opponent_talent_rating_of_losses | REAL |
| defense_strength_of_schedule | REAL |
| defense_wins_vs_favored_opponents | INTEGER |
| defense_losses_vs_weaker_opponents | INTEGER |
| defense_point_differential_vs_talent_expectation | REAL |
| created_at | TIMESTAMP |
| updated_at | TIMESTAMP |
|year| INTEGER|
|week| INTEGER|

# play_embedding

| Column | Data Type |
|--------|-----------|
| play_id | INTEGER |
| game_id | INTEGER |
| down | INTEGER |
| distance | REAL |
| yardline | REAL |
| yards_to_goal | REAL |
| period | INTEGER |
| clock | REAL |
| offense_score | REAL |
| defense_score | REAL |
| score_differential | REAL |
| offense_timeouts | REAL |
| defense_timeouts | REAL |
| is_rush | INTEGER |
| is_pass | INTEGER |
| is_punt | INTEGER |
| is_field_goal | INTEGER |
| is_extra_point | INTEGER |
| is_kickoff | INTEGER |
| is_penalty | INTEGER |
| is_timeout | INTEGER |
| is_sack | INTEGER |
| is_administrative | INTEGER |
| is_touchdown | INTEGER |
| is_completion | INTEGER |
| is_interception | INTEGER |
| is_fumble_lost | INTEGER |
| is_fumble_recovered | INTEGER |
| is_return_td | INTEGER |
| is_safety | INTEGER |
| is_good | INTEGER |
| is_two_point | INTEGER |
| created_at | TIMESTAMP |
| updated_at | TIMESTAMP |
| driveId | INTEGER |
| driveNumber | INTEGER |
| yardsGained | INTEGER |
| is_first_down | INTEGER |
|year| INTEGER|
|week| INTEGER|

# game_state_embedding

| Column | Data Type |
|--------|-----------|
| id | INTEGER |
| play_id | INTEGER |
| game_id | INTEGER |
| drive_id | INTEGER |
| drive_number | INTEGER |
| drive_plays_so_far | INTEGER |
| drive_yards_so_far | INTEGER |
| drive_start_yardline | INTEGER |
| drive_time_elapsed | INTEGER |
| down | INTEGER |
| distance | INTEGER |
| yardline | INTEGER |
| yards_to_goal | INTEGER |
| period | INTEGER |
| total_seconds_remaining | INTEGER |
| offense_score | INTEGER |
| defense_score | INTEGER |
| score_differential | INTEGER |
| offense_timeouts | INTEGER |
| defense_timeouts | INTEGER |
| venue_id | INTEGER |
| game_indoors | INTEGER |
| temperature | REAL |
| humidity | REAL |
| wind_speed | REAL |
| wind_direction | INTEGER |
| precipitation | REAL |
| is_field_turf | INTEGER |
| is_red_zone | INTEGER |
| is_goal_line | INTEGER |
| is_two_minute_warning | INTEGER |
| is_garbage_time | INTEGER |
| is_money_down | INTEGER |
| is_plus_territory | INTEGER |
| is_offense_home_team | INTEGER |
| conference_game | INTEGER |
| bowl_game | INTEGER |
| created_at | TIMESTAMP |
| updated_at | TIMESTAMP |
|year| INTEGER|
|week| INTEGER|

# play_targets

| Column | Data Type |
|--------|-----------|
| play_id | INT |
| game_id | INT |
| down | INT |
| distance | REAL |
| yardline | REAL |
| yards_to_goal | REAL |
| period | INT |
| clock | REAL |
| offense_score | REAL |
| defense_score | REAL |
| score_differential | REAL |
| offense_timeouts | REAL |
| defense_timeouts | REAL |
| is_rush | INT |
| is_pass | INT |
| is_punt | INT |
| is_field_goal | INT |
| is_extra_point | INT |
| is_kickoff | INT |
| is_penalty | INT |
| is_timeout | INT |
| is_sack | INT |
| is_administrative | INT |
| is_touchdown | INT |
| is_completion | INT |
| is_interception | INT |
| is_fumble_lost | INT |
| is_fumble_recovered | INT |
| is_return_td | INT |
| is_safety | INT |
| is_good | INT |
| is_two_point | INT |
| created_at | NUM |
| updated_at | NUM |
| driveId | INT |
| driveNumber | INT |
| yardsGained | INT |
| is_first_down | INT |
|year| INTEGER|
|week| INTEGER|

# drive_targets

| Column | Data Type |
|--------|-----------|
| drive_id | INT |
| driveNumber | INT |
| game_id | INT |
| startYardLine | INT |
| endYardLine | INT |
| totalYards |  |
| totalSeconds | REAL |
| playCount | INT |
| outcome_TD |  |
| outcome_FG |  |
| outcome_Punt |  |
| outcome_TurnoverOnDowns |  |
| outcome_Interception |  |
| outcome_Fumble |  |
| outcome_Safety |  |
| outcome_EndOfHalf |  |
| outcome_EndOfGame |  |
| scoringChangeOffense |  |
| scoringChangeDefense |  |
| outcome_MissedFG | INTEGER |
|year| INTEGER|
|week| INTEGER|

# game_targets

| Column | Data Type |
|--------|-----------|
| game_id | INTEGER |
| home_points | REAL |
| away_points | REAL |
| point_differential | REAL |
| total_points | REAL |
| season | INTEGER |
| week | INTEGER |
| season_type | TEXT |
| neutral_site | INTEGER |
| conference_game | INTEGER |
| attendance | REAL |
| home_start_elo | REAL |
| away_start_elo | REAL |
| home_rushing_yards | REAL |
| home_rushing_attempts | REAL |
| home_passing_yards | REAL |
| home_passing_attempts | REAL |
| home_passing_completions | REAL |
| away_rushing_yards | REAL |
| away_rushing_attempts | REAL |
| away_passing_yards | REAL |
| away_passing_attempts | REAL |
| away_passing_completions | REAL |
| home_yards_per_rush | REAL |
| home_yards_per_pass | REAL |
| home_yards_per_completion | REAL |
| away_yards_per_rush | REAL |
| away_yards_per_pass | REAL |
| away_yards_per_completion | REAL |
| home_passing_success_rate | REAL |
| home_rushing_success_rate | REAL |
| home_passing_explosiveness | REAL |
| home_rushing_explosiveness | REAL |
| away_passing_success_rate | REAL |
| away_rushing_success_rate | REAL |
| away_passing_explosiveness | REAL |
| away_rushing_explosiveness | REAL |
| home_explosive_play_count | REAL |
| home_explosive_play_rate | REAL |
| home_explosive_passing_count | REAL |
| home_explosive_rushing_count | REAL |
| away_explosive_play_count | REAL |
| away_explosive_play_rate | REAL |
| away_explosive_passing_count | REAL |
| away_explosive_rushing_count | REAL |
| created_at | TIMESTAMP |
| updated_at | TIMESTAMP |
|year| INTEGER|
|week| INTEGER|

Would Need Added:

  1. Complete data preprocessing pipeline with exact column mappings
  2. EmbeddingContainer class implementation
  3. Sequential data batching logic for TPU training
  4. Game state management system for play-by-play simulation
  5. Specific hyperparameter configurations
  6. Data joining logic across the 4 parquet tables
  7. Vegas data integration (where does closing line data come from?)


1) either B or C give me some pros and cons
2) I would like three layers i think? plays level drive level game level? coould be bidirectional... you have a better idea what would work best for this question than i do
3) final game stats, home score, away score, and scoring margin are the things that matter the most. really the game_targets is vital. it gives me two different ways to evaluate a game and its predictions. if the score isn't matching the stats projected i can see something fishy, or if i see the model expects good running number i can live bet better if i see the team stopping the run better than expected things like that. success for the model is predicting within an RMSE of like 12-13 as i believe vegas' RMSE is 14 and some change and all of my game level models were 15+
4) yes those test splits. no holdouts, 2021 should matter more than the other training seasons and 2020 should matter the least if they need to be weighted but i think keeping all things equal is best to start
5) i think i want it to take <48 hours to train. unless there are ways to train in like 12 hour bunches and the model keep everything in mind. I just cannot have longer than 12 hour training sessions without restarting.
6) state consistency because eventually this will get put into a monte carlo system so the better sims it gives is important
