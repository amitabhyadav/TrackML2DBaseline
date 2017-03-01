__author__ = 'mikhail91'


import numpy

class NearestHit(object):


    def __init__(self, min_cos_value=0.9):
        """
        Track Pattern Recognition based on the connections between two nearest hits from two nearest detector layers.

        Parameters
        ----------
        min_cos_value : float
            Minimum cos value between two nearest segments of the track.
        """

        self.min_cos_value = min_cos_value

    def fit(self, X, y):
        pass


    def predict_one_event(self, X):
        """
        Track Pattern Recognition for one event.

        Parameters
        ----------
        X : ndarray-like
            Hit features.

        Returns
        -------
        labels : array-like
            Recognized track labels.
        """

        x, y, layers = X[:, 3], X[:, 4], X[:, 1]
        used = numpy.zeros(len(x))
        labels = -1. * numpy.ones(len(x))
        track_id = 0

        # Start from a hit in the first layer
        for first_id in numpy.arange(0, len(x))[layers == 0]:

            track = []
            track += [first_id]
            used[first_id] = 1

            # Go through other detector layers
            for one_layer in numpy.unique(layers)[1:]:

                # Select hits of the layer
                hit_ids = numpy.arange(0, len(x))[(layers == one_layer)*(used == 0)]

                # Compute distance between hits
                diff_r = (x[track[-1]] - x[hit_ids])**2 + (y[track[-1]] - y[hit_ids])**2

                if len(diff_r) == 0:
                    break

                # Select new track hit
                track_hit_id = hit_ids[diff_r == diff_r.min()][0]

                # Check cos of angle between two track segments
                if one_layer != 1:

                    x1, x2, x3 = x[track[-2]],  x[track[-1]], x[track_hit_id]
                    y1, y2, y3 = y[track[-2]],  y[track[-1]], y[track_hit_id]
                    dx1, dx2 = x2 - x1, x3 - x2
                    dy1, dy2 = y2 - y1, y3 - y2

                    cos = (dx1*dx2 + dy1*dy2) / numpy.sqrt((dx1**2 + dy1**2) * (dx2**2 + dy2**2))

                    if cos < self.min_cos_value:
                        break

                # Add new hit to the track
                track += [track_hit_id]
                used[track_hit_id] = 1

            # Label the track hits
            labels[track] = track_id
            track_id += 1


        return labels

    def predict(self, X):
        """
        Track Pattern Recognition for all event.

        Parameters
        ----------
        X : ndarray-like
            Hit features.

        Returns
        -------
        labels : array-like
            Recognized track labels.
        """

        event_ids = numpy.unique(X[:, 0])
        labels = []

        for one_event_id in event_ids:

            X_event = X[X[:, 0] == one_event_id]
            labels_event = self.predict_one_event(X_event)
            labels += list(labels_event)

        return numpy.array(labels)
