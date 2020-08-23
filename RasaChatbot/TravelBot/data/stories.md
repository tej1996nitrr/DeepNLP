## happy path
* greet
  - utter_greet
* mood_great
  - utter_happy

## sad path 1
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* affirm
  - utter_happy

## sad path 2
* greet
  - utter_greet
* mood_unhappy
  - utter_cheer_up
  - utter_did_that_help
* deny
  - utter_goodbye

## say goodbye
* goodbye
  - utter_goodbye

## bot challenge
* bot_challenge
  - utter_iamabot
 
## story 1.1 greet+search_flights+thanks
* greet
  - utter_greet
* search_flights{'source':'bangalore','destination':'mumbai'}
  - action_find_flights
* thanks
  - utter_welcome
 
## story 1.2 greet+search_flights+thanks
* greet
  - utter_greet
* search_flights{'source':'lucknow','destination':'delhi'}
  - action_find_flights
* thanks
  - utter_welcome
 
## story 2.1 greet+search_flights+bye
* greet
  - utter_greet
* search_flights{'source':'lucknow','destination':'delhi'}
  - action_find_flights
* bye
  - utter_goodbye
 
## story 2.2 greet+search_flights+bye
* greet
  - utter_greet
* search_flights{'source':'bangalore','destination':'mumbai'}
  - action_find_flights
* bye
  - utter_goodbye
 
## story 2.3 greet+search_flights+bye
* greet
  - utter_greet
* search_flights{'source':'bangalore','destination':'hyderabad'}
  - action_find_flights
* bye
  - utter_goodbye

## story 3.1 greet+search_flights+thanks
* greet
  - utter_greet
* search_flights{'source':'raipur','destination':'mumbai'}
  - action_find_flights
* thanks
  - utter_welcome

## story 3.2 greet+search_flights+thanks
* greet
  - utter_greet
* search_flights{'source':'pune','destination':'mumbai'}
  - action_find_flights
* thanks
  - utter_welcome
  
## story 4.1 greet+search_trains+thanks
* greet
  - utter_greet
* search_trains{'source':'raipur','destination':'mumbai'}
  - action_find_trains
* thanks
  - utter_welcome

## story 3.2 greet+search_trains+thanks
* greet
  - utter_greet
* search_trains{'source':'pune','destination':'mumbai'}
  - action_find_trains
* thanks
  - utter_welcome
 
## story 5.1 greet+search_trains+bye
* greet
  - utter_greet
* search_trains{'source':'lucknow','destination':'delhi'}
  - action_find_trains
* bye
  - utter_goodbye
 
## story 5.2 greet+search_trains+bye
* greet
  - utter_greet
* search_trains{'source':'bangalore','destination':'mumbai'}
  - action_find_trains
* bye
  - utter_goodbye
 
## story 5.3 greet+search_trains+bye
* greet
  - utter_greet
* search_trains{'source':'bangalore','destination':'hyderabad'}
  - action_find_trains
* bye
  - utter_goodbye
  
## fallback
  - utter_unclear

## story 7 greet+find_itineraries+thanks
* greet
  - utter_greet
* find_itineraries
  - action_find_itineraries
* thanks
  - utter_welcome

## story 8 greet+find_itineraries+thanks+bye
* greet
  - utter_greet
* find_itineraries
  - action_find_itineraries
* thanks
  - utter_welcome
* bye
  - utter_goodbye
  
## story 9 greet+find_itineraries+bye
* greet
  - utter_greet
* find_itineraries
  - action_find_itineraries
***** bye
  -utter_goodbye