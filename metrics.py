__author__ = 'mikhail91'

import numpy
import pandas


class HitsMatchingEfficiency(object):
    def __init__(self, eff_threshold=0.5, min_hits_per_track=1):

        self.eff_threshold = eff_threshold
        self.min_hits_per_track = min_hits_per_track

    def fit(self, true_labels, track_inds):

        track_efficiencies = []
        track_labels = []
        reco_eff = 0
        n_ghosts = 0
        n_tracks = 0

        n_tracks = len(numpy.unique(true_labels))

        for one_track in track_inds:

            one_track = numpy.unique(one_track)

            if len(one_track) < self.min_hits_per_track:
                one_track_eff = 0
                track_efficiencies.append(one_track_eff)
                continue

            hits_labels = true_labels[one_track]
            unique_hits_labels, count_hits_labels = numpy.unique(hits_labels, return_counts=True)
            one_track_eff = 1. * count_hits_labels.max() / len(one_track)
            one_track_label = unique_hits_labels[count_hits_labels == count_hits_labels.max()]

            track_efficiencies.append(one_track_eff)

            if one_track_eff >= self.eff_threshold:
                track_labels.append(one_track_label)
            else:
                n_ghosts += 1


        self.efficiencies_ = track_efficiencies
        if len(track_efficiencies) == 0:
            self.avg_efficiency_ = 0
        else:
            self.avg_efficiency_ = numpy.array(track_efficiencies).mean()


        if n_tracks == 0:
            self.ghost_rate_ = 0
        else:
            self.ghost_rate_ = 1. * n_ghosts / n_tracks


        if n_tracks == 0 or len(track_labels) == 0:
            self.reconstruction_efficiency_ = 0
        else:
            self.reconstruction_efficiency_ = 1. * len(numpy.unique(track_labels)) / n_tracks


        if n_tracks == 0 or len(track_labels) == 0:
            self.clone_rate_ = 0
        else:
            self.clone_rate_ = 1. * (len(track_labels) - len(numpy.unique(track_labels))) / n_tracks



class RecognitionQuality(object):

    def __init__(self, real_tracks, recognized_tracks, track_eff_threshold, min_hits_per_track):
        """
        This class is used to evaluate tracks recognition quality for all events.

        Parameters
        ----------
        real_tracks : pandas.DataFrame
            Events and tracks.
        recognized_tracks : pandas.DataFrame
            Recognized tracks for the events.
        track_eff_threshold : float
            Track Finding Efficiency threshold.
        min_hits_per_track : int
            Minimum number of hits per track.

        Return
        ------
        report_event : pandas.DataFrame
            Track recognition quality for all events.
        report_tracks : pandas.DataFrame
            Track recognition quality for all tracks in all events.
        """

        self.real_tracks = real_tracks
        self.recognized_tracks = recognized_tracks
        self.track_eff_threshold = track_eff_threshold
        self.min_hits_per_track = min_hits_per_track

        self.report_ = None

    def calculate(self):

        reco_eff = []
        ghost_rate = []
        clone_rate = []
        mean_track_eff = []
        event_ids_col = []

        track_eff = []
        evnt_ids_col2 = []
        track_ids = []

        event_ids = numpy.unique(self.real_tracks.event.values)

        for one_event_id in event_ids:

            real_event = self.real_tracks[self.real_tracks.event.values == one_event_id]
            real_event['hit_global_index'] = real_event.index
            real_event.reset_index(drop=True, inplace=True)
            reco_event = self.recognized_tracks[self.recognized_tracks.event.values == one_event_id]
            reco_event.reset_index(drop=True, inplace=True)

            hit2ind = dict(zip(real_event['hit_global_index'].values, real_event.index.values))

            track_inds = []

            for reco_track_id in numpy.unique(reco_event.track.values):

                reco_track_hits = reco_event['hit_index'].values[reco_event.track.values == reco_track_id]
                reco_track_inds = [hit2ind[i] for i in reco_track_hits]

                track_inds.append(reco_track_inds)

            track_inds = numpy.array(track_inds)


            hme = HitsMatchingEfficiency(eff_threshold=self.track_eff_threshold, min_hits_per_track=self.min_hits_per_track)
            hme.fit(true_labels=real_event.particle.values, track_inds=track_inds)

            reco_eff += [hme.reconstruction_efficiency_]
            ghost_rate += [hme.ghost_rate_]
            clone_rate += [hme.clone_rate_]
            mean_track_eff += [numpy.mean(hme.efficiencies_)]
            event_ids_col += [one_event_id]

            track_ids += list(numpy.unique(reco_event.track.values))
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