__author__ = 'mikhail91'

import numpy
from sklearn.linear_model import LinearRegression

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
