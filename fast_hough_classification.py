__author__ = 'mikhail91'


import numpy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier


class TrackRegression(object):

    def __init__(self):
        """
        Tracks regression
        """

        self.model = None

        self.theta_ = None
        self.invr_ = None

    def fit(self, x, y, sample_weight=None):
        """
        Fit track regression model:
        r = 2 * r0 * Cos(theta) *Cos(phi) + 2 * r0 * Sin(theta) * Sin(phi)
        where 1. / r0 and theta are track parameters.

        Parameters
        ----------
        x : array_like
            X-coordinates of hits.
        y : array_like
            Y-coordinates of hits.
        sample_weight : array_like
            Weights of hits.
        """

        x = numpy.array(x)
        y = numpy.array(y)
        if sample_weight != None:
            sample_weight = numpy.array(sample_weight)

        phi = numpy.arctan(y / x) * (x != 0) + numpy.pi * (x < 0) + 0.5 * numpy.pi * (x==0) * (y>0) + 1.5 * numpy.pi * (x==0) * (y<0)
        r = numpy.sqrt(x**2 + y**2)

        xx1, xx2 = numpy.cos(phi), numpy.sin(phi)
        yy = r

        XX = numpy.concatenate((xx1.reshape(-1, 1), xx2.reshape(-1, 1)), axis=1)

        lr = LinearRegression(fit_intercept=False, copy_X=False)
        lr.fit(XX, yy, sample_weight)

        a, b = lr.coef_
        theta = numpy.arctan(b / a) * (a != 0) + numpy.pi * (a < 0) + 0.5 * numpy.pi * (a==0) * (b>0) + 1.5 * numpy.pi * (a==0) * (b<0)
        r0 = numpy.sqrt(a**2 + b**2) / 2.

        self.theta_ = theta
        self.invr_ = 1. / r0
        self.model = lr

    def predict(self, phi):
        """
        Predict using the track regression model:
        r = 2 * r0 * Cos(theta) *Cos(phi) + 2 * r0 * Sin(theta) * Sin(phi)
        where 1. / r0 and theta are track parameters.

        Parameters
        ----------
        phi : array_like
            Array of hit phis.

        Return
        ------
        x_pred : array_like
            Predicted x-coordinates of hits.
        y_pred : array_like
            Predicted y-coordinates of hits.
        """

        phi = numpy.array(phi)

        if self.model == None:
            print "Fit the model first."
            return

        #phi = numpy.arctan(y / x)
        xx1, xx2 = numpy.cos(phi), numpy.sin(phi)
        XX = numpy.concatenate((xx1.reshape(-1, 1), xx2.reshape(-1, 1)), axis=1)

        r_predict = self.model.predict(XX)

        y_pred = r_predict * numpy.sin(phi)
        x_pred = r_predict * numpy.cos(phi)

        return x_pred, y_pred


def get_track_efficiency(true_labels, reco_labels):

    true_labels = numpy.array(true_labels)
    reco_labels = numpy.array(reco_labels)

    unique, counts = numpy.unique(true_labels, return_counts=True)

    if len(reco_labels) == 0:
        eff = 0.
    else:
        eff = 1. * counts.max() / len(reco_labels)

    return eff



class Clusterer(object):

    def __init__(self, invr_size=0.05 / 20000., theta_size=2. * numpy.pi / 450.,  min_hits=2, phi_window=0.5, classifier=None, proba_threshold=0.5):
        """
        Track pattern recognition for one event based on Hough Transform.

        Parameters
        ----------
        invr_size : float
            Bin width along 1/r0 axis of track parameters space.
        theta_size : float
            Bin width along theta axis of track parameters space.
        min_hits : int
            Minimum number of hits per on recognized track.
        phi_window : float
            Hits phi window width.
        """

        self.invr_size = invr_size
        self.theta_size = theta_size
        self.min_hits = min_hits
        self.phi_window = phi_window
        self.classifier = classifier
        self.proba_threshold = proba_threshold


    def get_polar(self, x, y):
        """
        Calculate hits phi coordinates in polar system.

        Parameters
        ----------
        x : array-like
            X-coordinates of hits.
        y : array-like
            Y-coordinates of hits.

        Returns
        -------
        phi : array-like
            Phi coordinates of hits.
        """

        x = numpy.array(x)
        y = numpy.array(y)

        phi = numpy.arctan(y / x) * (x != 0) + numpy.pi * (x < 0) + 0.5 * numpy.pi * (x==0) * (y>0) + 1.5 * numpy.pi * (x==0) * (y<0)
        r = numpy.sqrt(x**2 + y**2)

        return r, phi

    def hits_in_bin(self, r, phi, invr_bin, theta_bin, phi_min, phi_max):
        """
        Estimate hits inside a bin of the tracks parameters.

        Parameters
        ----------
        r : array-like
            R coordinates of hits.
        phi : array_like
            Phi coordinates of hits.
        invr_bin : float
            1/r coordinate of a bin center.
        theta_bin : float
            Theta coordinate of a bin center.
        phi_min : float
            Min hit phi value.
        phi_max : float
            Max hit phi value.

        Returns
        -------
        track_inds : array_like
            List of hit indexes inside the bin.
        """

        inds = numpy.arange(0, len(r))

        invr_left = 2 * numpy.cos(phi - theta_bin + self.theta_size) / r
        invr_right = 2 * numpy.cos(phi - theta_bin - self.theta_size) / r


        sel = (invr_left >= invr_bin - 0.5 * self.invr_size) * (invr_right <= invr_bin + 0.5 * self.invr_size) + \
              (invr_left <= invr_bin + 0.5 * self.invr_size) * (invr_right >= invr_bin - 0.5 * self.invr_size)
        sel = sel * (phi >= phi_min) * (phi <= phi_max)

        track_inds = inds[sel]

        return track_inds


    def tranform(self, x, y, layer):
        """
        Hough Transformation and tracks pattern recognition.

        Parameters
        ----------
        x : array_like
            X-coordinates of hits.
        y : array_like
            Y-coordinates of hits.
        layer : array_like
            Layer numbers of hits.

        Return
        ------
        track_inds : ndarray
            List of recognized tracks. Each track is a list of its hit indexes.
        track_params : ndarray
            List of track parameters.
        """

        track_inds = []
        track_params = []

        r, phi = self.get_polar(x, y)

        # Go through pairs of hits
        for first_i in range(len(r)-1):

            r1, phi1, layer1 = r[first_i], phi[first_i], layer[first_i]

            for second_i in range(first_i+1, len(r)):

                r2, phi2, layer2 = r[second_i], phi[second_i], layer[second_i]

                # Take hits from different layers if a phi window to speed up the method
                if numpy.abs(layer2 - layer1) == 0 or numpy.abs(phi1 - phi2) > self.phi_window:
                    continue

                # Estimate track parameters based on pair of hits
                b, a = (numpy.cos(phi2) * r1 / r2 - numpy.cos(phi1)), (numpy.sin(phi1) - numpy.sin(phi2) * r1 / r2)
                theta = numpy.arctan(b / a) * (a != 0) + numpy.pi * (a < 0) + 0.5 * numpy.pi * (a==0) * (b>0) + 1.5 * numpy.pi * (a==0) * (b<0)
                invr = 2. * numpy.cos(phi1 - theta) / r1

                # Estimate hits inside a bin of the track parameters space
                track = self.hits_in_bin(r, phi, invr, theta, 0.5 * (phi1 + phi2 - self.phi_window), 0.5 * (phi1 + phi2 + self.phi_window))

                # Save recognized track
                if len(track) >= self.min_hits:
                    track_inds.append(track)
                    track_params.append([invr, theta])

        track_inds = numpy.array(track_inds)
        track_params = numpy.array(track_params)



        return track_inds, track_params


    def get_hit_labels(self, track_inds, n_hits):
        """
        Estimate hit labels based on the recognized tracks.

        Parameters
        ----------
        track_inds : ndarray
            List of recognized tracks. Each track is a list of its hit indexes.
        n_hits : int
            Number of hits in the event.

        Return
        ------
        labels : array-like
            Hit labels.
        """

        labels = -1. * numpy.ones(n_hits)
        used = numpy.zeros(n_hits)
        track_id = 0

        counter = 0


        while 1:

            track_lens = numpy.array([len(i[used[i] == 0]) for i in track_inds])

            if len(track_lens) == 0:
                break

            max_len = track_lens.max()

            if max_len < self.min_hits:
                break

            one_track_inds = track_inds[track_lens == track_lens.max()][0]
            one_track_inds = one_track_inds[used[one_track_inds] == 0]

            used[one_track_inds] = 1
            labels[one_track_inds] = track_id
            track_id += 1

        return numpy.array(labels)


    def splitter(self, labels, X):
        """
        Separate two close tracks.

        Parameters
        ----------
        labels : array-like
            Recognized hit labels.
        X : ndarray-like
            Hit features.

        Returns
        ------
        labels : array-like
            New recognized hit labels.
        """

        x, y, layer = X[:, 2], X[:, 3], X[:, 0]
        r, phi = self.get_polar(x, y)

        ind = numpy.arange(len(X))
        unique_labels = numpy.unique(labels[labels != -1])
        track_id = unique_labels[-1] + 1
        #print unique_labels

        for lab in unique_labels:

            track_ind = ind[labels == lab]

            track_layer = layer[track_ind]
            track_phi = phi[track_ind]

            track1 = []
            track2 = []

            for l in numpy.unique(track_layer):

                ind_layer = track_ind[track_layer == l]
                phi_layer = track_phi[track_layer == l]

                hit_loc_ind = numpy.argsort(phi_layer)

                track1.append(ind_layer[hit_loc_ind[0]])

                if len(ind_layer) > 1:

                    track2.append(ind_layer[hit_loc_ind[-1]])

            if len(track2)>=2:
                labels[track2] = track_id
                track_id += 1

        return labels

    def marker(self, labels, X):
        """
        Marks unlabeled hits.

        Parameters
        ----------
        labels : array-like
            Recognized hit labels.
        X : ndarray-like
            Hit features.

        Returns
        ------
        labels : array-like
            New recognized hit labels.
        """

        x, y, layer = X[:, 2], X[:, 3], X[:, 0]
        r, phi = self.get_polar(x, y)

        ind = numpy.arange(len(X))

        unlabeled_hits = ind[(labels == -1)*(layer == 0)]
        l1_hits = ind[(layer == 1)]

        for hit_ind in unlabeled_hits:

            dist = (x[hit_ind] - x[l1_hits])**2 + (y[hit_ind] - y[l1_hits])**2
            nearest_hit_ind = l1_hits[dist == dist.min()][0]

            labels[hit_ind] = labels[nearest_hit_ind]

        return labels


    def track_features(self, x, y):
        """
        Calculate track feature values for a classifier.

        Parameters
        ----------
        X : ndarray-like
            Hit features.

        Returns
        -------
        n_hits : int
            Number of hits of a track.
        theta : float
            Theta parameter of a track.
        invr : float
            1/r0 parameter of a track.
        rmse : float
            RMSE of a track fit.
        """

        #x, y = X[:, 3], X[:, 4]

        # Number of hits of a track
        n_hits = len(x)

        # Fit track parameters
        tr = TrackRegression()
        tr.fit(x, y)

        theta, invr = tr.theta_, tr.invr_


        # Predict hit coordinates of the fitted track
        r, phi = self.get_polar(x, y)
        x_pred, y_pred = tr.predict(phi)

        # Calculate RMSE
        rmse = numpy.sqrt(((y - y_pred)**2).sum())

        return n_hits, theta, invr, rmse


    def new_track_inds(self, track_inds, X, classifier):
        """
        Reduces number of ghosts among recognized tracks using trained classifier.

        Parameters
        ----------
        track_inds : ndarray-like
            Array of recognized track indexes.
        X : ndarray-like
            Hit features.
        classifier : object
            Trained classifier.

        Returns
        -------
        new_track_inds : ndarray-like
            Array of new track indexes.
        """

        if classifier == None:
            return track_inds

        # Generate data for the classifier
        XX = []

        for track in track_inds:

            features = self.track_features(X[track,2], X[track,3])
            XX.append(features)

        XX = numpy.array(XX)


        # Predict probability to be a good track
        labels_pred = classifier.predict_proba(XX)[:, 1]


        # Select good tracks based on the probability
        track_inds = numpy.array(track_inds)
        new_track_inds = track_inds[labels_pred >= self.proba_threshold] # Larger the threshold value, better the ghosts reduction

        return new_track_inds



    def fit(self, X, y):

        if self.classifier == None:
            return


        XX = []
        yy = []

        # Create sample to train the classifier
        event_ids = numpy.unique(X[:, 0])

        for one_event_id in event_ids:

            # Select one event
            X_event = X[X[:, 0] == one_event_id]
            y_event = y[X[:, 0] == one_event_id]

            # Get recognized track inds. Approach: one hit can belong to several tracks
            track_inds, track_params = self.tranform(X_event[:, 3], X_event[:, 4], X_event[:, 1])

            # Calculate feature values for the recognized tracks
            for track in track_inds:

                # Select one recognized track
                X_track = X_event[track, :]
                y_track = y_event[track]

                # Calculate feature values for the track
                features = self.track_features(X_track[:, 3], X_track[:, 4])
                XX.append(features)

                # Calculate the track true labels in {0, 1}. 0 - ghost, 1 - good track.
                eff = get_track_efficiency(y_track[:, 1], [1]*len(y_track))
                label = (eff >= 0.8) * 1.

                yy += [label]

        XX = numpy.array(XX)
        yy = numpy.array(yy)

        # Balance {0, 1} classes. This improves the classification quality.
        weights = numpy.zeros(len(yy))
        weights += 1. * len(yy) / len(yy[yy == 0]) * (yy == 0) + \
                   1. * len(yy) / len(yy[yy == 1]) * (yy == 1)

        # Train the classifier
        self.classifier.fit(XX, yy, weights)

        return XX, yy, weights




    def predict_single_event(self, X):
        """
        Hough Transformation and tracks pattern recognition for one event.

        Parameters
        ----------
        X : ndarray_like
            Hit features.

        Return
        ------
        Labels : array-like
            Track id labels for the each hit.
        """

        x, y, layer = X[:, 2], X[:, 3], X[:, 0]

        if self.classifier == None:
            track_inds, track_params = self.tranform(x, y, layer)
        else:
            track_inds_origin, track_params = self.tranform(x, y, layer)
            track_inds = self.new_track_inds(track_inds_origin, X, self.classifier)


        self.track_inds_ = track_inds
        self.track_params_ = track_params

        # Assign one track label to hits
        labels = self.get_hit_labels(track_inds, len(X))

        # Additional processing of the recognized labels (+1-2% to score)
        labels = self.splitter(labels, X)
        labels = self.marker(labels, X)

        return labels
