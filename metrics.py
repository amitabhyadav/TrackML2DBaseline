__author__ = 'mikhail91'

import numpy
import pandas


class HitsMatchingEfficiency(object):
    def __init__(self, eff_threshold=0.5, min_hits_per_track=1):
        """
        This class calculates track efficiencies, reconstruction efficiency, ghost rate and clone rate
        for one event using hits matching approach.

        Parameters
        ----------
        eff_threshold : float
            Threshold value of a track efficiency to consider the track as a reconstructed one.
        min_hits_per_track : int
            Minimum number of hit per one recognized track.
        """

        self.eff_threshold = eff_threshold
        self.min_hits_per_track = min_hits_per_track

    def fit(self, true_labels, reco_labels):
        """
        The method calculates the metrics.

        Parameters
        ----------
        true_labels : array-like
            True hit labels.
        reco_labels : array-like
            Recognized hit labels.
        """

        true_labels = numpy.array(true_labels)
        reco_labels = numpy.array(reco_labels)


        unique_labels = numpy.unique(reco_labels)

        # Calculate efficiencies
        efficiencies = []
        tracks_id = []

        for label in unique_labels:
            if label != -1:
                track = true_labels[reco_labels == label]
                # if len(track[track != -1]) == 0:
                #    continue
                unique, counts = numpy.unique(track, return_counts=True)

                if len(track) >= self.min_hits_per_track:
                    eff = 1. * counts.max() / len(track)
                else:
                    eff = 0.
                efficiencies.append(eff)

                tracks_id.append(unique[counts == counts.max()][0])

        tracks_id = numpy.array(tracks_id)
        efficiencies = numpy.array(efficiencies)
        self.efficiencies_ = efficiencies

        # Calculate avg. efficiency
        avg_efficiency = efficiencies.mean()
        self.avg_efficiency_ = avg_efficiency

        # Calculate reconstruction efficiency
        true_tracks_id = numpy.unique(true_labels)
        n_tracks = (true_tracks_id != -1).sum()

        reco_tracks_id = tracks_id[efficiencies >= self.eff_threshold]
        unique, counts = numpy.unique(reco_tracks_id[reco_tracks_id != -1], return_counts=True)

        if n_tracks > 0:
            reconstruction_efficiency = 1. * len(unique) / (n_tracks)
            self.reconstruction_efficiency_ = reconstruction_efficiency
        else:
            self.reconstruction_efficiency_ = 0

        # Calculate ghost rate
        if n_tracks > 0:
            ghost_rate = 1. * (len(tracks_id) - len(reco_tracks_id[reco_tracks_id != -1])) / (n_tracks)
            self.ghost_rate_ = ghost_rate
        else:
            self.ghost_rate_ = 0

        # Calculate clone rate
        reco_tracks_id = tracks_id[efficiencies >= self.eff_threshold]
        unique, counts = numpy.unique(reco_tracks_id[reco_tracks_id != -1], return_counts=True)

        if n_tracks > 0:
            clone_rate = (counts - numpy.ones(len(counts))).sum() / (n_tracks)
            self.clone_rate_ = clone_rate
        else:
            self.clone_rate_ = 0



class RecognitionQuality(object):

    def __init__(self, track_eff_threshold, min_hits_per_track):
        """
        This class is used to evaluate tracks recognition quality for all events.

        Parameters
        ----------
        track_eff_threshold : float
            Track Finding Efficiency threshold.
        min_hits_per_track : int
            Minimum number of hits per track.
        """

        self.track_eff_threshold = track_eff_threshold
        self.min_hits_per_track = min_hits_per_track

    def calculate(self, y, y_reco):
        """
        Return
        ------
        X : ndarray-like
            Hit features.
        y : array-like
            True hit labels.
        y_reco : array-like
            Reconstructed hit labels.
        """

        reco_eff = []
        ghost_rate = []
        clone_rate = []
        mean_track_eff = []
        event_ids_col = []

        track_eff = []
        evnt_ids_col2 = []
        track_ids = []

        event_ids = numpy.unique(y[:, 0])

        for one_event_id in event_ids:

            true_labels = y[y[:, 0] == one_event_id, 1]
            reco_labels = y_reco[y[:, 0] == one_event_id]


            hme = HitsMatchingEfficiency(eff_threshold=self.track_eff_threshold, min_hits_per_track=self.min_hits_per_track)
            hme.fit(true_labels=true_labels, reco_labels=reco_labels)

            reco_eff += [hme.reconstruction_efficiency_]
            ghost_rate += [hme.ghost_rate_]
            clone_rate += [hme.clone_rate_]
            mean_track_eff += [numpy.mean(hme.efficiencies_)]
            event_ids_col += [one_event_id]

            track_ids += list(numpy.unique(reco_labels[reco_labels != -1]))
            track_eff += list(hme.efficiencies_)
            evnt_ids_col2 += [one_event_id] * len(hme.efficiencies_)



        report_events = pandas.DataFrame()
        report_events['Event'] = event_ids_col
        report_events['ReconstructionEfficiency'] = reco_eff
        report_events['GhostRate'] = ghost_rate
        report_events['CloneRate'] = clone_rate
        report_events['AvgTrackEfficiency'] = mean_track_eff

        report_tracks = pandas.DataFrame()
        report_tracks['Event'] = evnt_ids_col2
        report_tracks['Track'] = track_ids
        report_tracks['TrackEfficiency'] = track_eff

        return report_events, report_tracks