__author__ = 'mikhail91'


import numpy

class Clusterer(object):

    def __init__(self, invr_size=0.05 / 20000., theta_size=2. * numpy.pi / 450.,  min_hits=2, phi_window=0.5):
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

    def fit(self, X, y):
        pass

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
        track_inds, track_params = self.tranform(x, y, layer)

        self.track_inds_ = track_inds
        self.track_params_ = track_params

        # Assign one track label to hits
        labels = self.get_hit_labels(track_inds, len(X))

        # Additional processing of the recognized labels (+1-2% to score)
        labels = self.splitter(labels, X)
        labels = self.marker(labels, X)

        return labels
