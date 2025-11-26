import datetime as dt
import random
from sgp4.conveniences import jday

from tle_propagator.time import Epoch

def test():
    # Generate 10 random dates
    dates = [dt.datetime.fromtimestamp(
        random.uniform(dt.datetime(1957, 10, 4).timestamp(), 
                       dt.datetime(2025, 12, 31).timestamp())               
        ) for _ in range(10)]
    
    for date in dates:
        # JD and fraction from calendar date
        jd, fr = jday(date.year, date.month, date.day, date.hour, date.minute, date.second) # Use sgp4 convenience function to convert to JD and fraction
        print(f"Testing date: {date}, JD: {jd}, Fraction: {fr}")
        epoch_from_jd = Epoch(jd, fr)
        assert abs((epoch_from_jd.jd + epoch_from_jd.fr) - (jd + fr)) < 1e-8, "JD to Epoch conversion error."
        Y, m, d, H, M, S = epoch_from_jd.calendar
        assert (Y, m, d, H, M, S) == (date.year, date.month, date.day, date.hour, date.minute, date.second)

        # Conversion to calendar date
        epoch_from_calendar = Epoch.from_calendar(date.year, date.month, date.day, date.hour, date.minute, date.second)
        assert abs((epoch_from_calendar.jd + epoch_from_calendar.fr) - (jd + fr)) < 1e-8, "Calendar to JD conversion error."