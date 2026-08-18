import pandas as pd
import statsapi

print(statsapi.player_stats(next(x['id'] for x in statsapi.get('sports_players',{'season':2008,'gameType':'W'})
['people'] if x['fullName']=='Chase Utley'), 'hitting', 'career'))