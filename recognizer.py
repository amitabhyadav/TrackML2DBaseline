__author__ = 'mikhail91'

import pandas
import numpy

class TrackRecognizer(object):

    def __init__(self, method):
        """
        This class is used to recognize tracks in several events.

        Parameters
        ----------
        model : obj
            Method for one event tracks recognition.
        """

        self.method = method

    def predict(self, events):
        """
        This method recognizes tracks in all events.

        Parameters
        ----------
        events : pandas.DataFrame
            Tracks of all events.

        Return
        ------
        results : pandas.DataFrame
            Recognized tracks for all events.
        """

        event = []
        track = []
        hit = []
        x = []
        y = []

        event_ids = numpy.unique(events.event.values)

        for one_event_id in event_ids:

            one_event = events[events.event.values == one_event_id]

            self.method.fit(one_event[['x', 'y']].values, one_event.particle.values)
            labels = self.method.predict(one_event[['x', 'y']].values)

            event += [one_event_id] * len(labels)
            track += list(labels)
            x += list(one_event.x.values)
            y += list(one_event.y.values)



        results = pandas.DataFrame()
        results['event'] = event
        results['track'] = track
        results['x'] = x
        results['y'] = y

        return results
