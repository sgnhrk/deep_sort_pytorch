# vim: expandtab:ts=4:sw=4
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    Fullyoccluded は画面から消えて検出結果がなくなった状態だが，トラッキングとしては
    継続状態にしたい状態のこと．_partially_occluded = true かつ static = true の状態で，
    missed を実行されるとこの状態に入る．
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    Fullyoccluded = 4


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    static_thresh : float
    static_set_frames : int
        bbox の下端中点の変動が static_thresh 以下であり，それが static_set_frames 続いたら
        その track に static フラグを付ける．static_set_frames が 0 の場合，static フラグは付けない．
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, static_thresh=5.0, static_set_frames=0, static_unset_frames=10, occluded=False, max_age_fully_occluded=0):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self.static = False
        self.partially_occluded = occluded
        self.static_count = 0
        self.static_unset_count = 0
        self.tlbr_in_prev_frame = None

        self._static_tlwh = None
        self._n_init = n_init
        self._max_age = max_age
        self._static_thresh = static_thresh
        self._static_set_frames = static_set_frames
        self._static_unset_frames = static_unset_frames
        self._max_age_fully_occluded = max_age_fully_occluded

        #self._d_hist = np.zeros(10, dtype=np.float32)

    def to_xyah(self):
        '''Get current position in bounding box format (bbox center x, center y, aspect, height)
        '''
        return self.mean[:4].copy()

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        if self.state != TrackState.Fullyoccluded:
            self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.time_since_update += 1
        self.age += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """

        if self.static and not self.partially_occluded and detection.occluded:
            self._static_tlwh = self.to_tlwh()

        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.partially_occluded = detection.occluded

        # static object の判定
        if self._static_set_frames > 0:

            tlbr = self.to_tlbr()
            if self.tlbr_in_prev_frame is not None:
                x = (tlbr[0] + tlbr[2]) / 2
                y = tlbr[3]
                px = (self.tlbr_in_prev_frame[0] + self.tlbr_in_prev_frame[2]) / 2
                py = self.tlbr_in_prev_frame[3]
                d = np.sum( (x-px)**2 + (y-py)**2 )
                d = np.sqrt(d)
                thresh = self._static_thresh
                #print(self.track_id, x, y, px, py, d, thresh)
                if d < thresh:
                    self.static_count = min(self.static_count + 1, self._static_set_frames)
                    self.static_unset_count = 0
                elif self.partially_occluded == False:
                    self.static_count = 0
                    self.static_unset_count = min(self.static_unset_count + 1, self._static_unset_frames)

            self.tlbr_in_prev_frame = tlbr

            if self.static == False:
                if (self.static_count == self._static_set_frames):
                    self.static = True
            else:
                if self.partially_occluded == False and self.static_count == 0:
                    if (self.static_unset_count == self._static_unset_frames):
                        self.static = False

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        elif self.state == TrackState.Fullyoccluded:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.partially_occluded and self.static and self.state == TrackState.Confirmed:
            self.state = TrackState.Fullyoccluded
        elif self.state == TrackState.Fullyoccluded:
            if self.time_since_update > self._max_age_fully_occluded:
                self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def is_static(self):
        return self.static

    def is_fully_occluded(self):
        return self.state == TrackState.Fullyoccluded

    def is_static_and_fully_occluded(self):
        return self.is_static() and self.is_fully_occluded()

    def is_partially_occluded(self):
        return self.partially_occluded and not self.is_fully_occluded()

    def to_static_tlwh(self):
        if self._static_tlwh is None:
            return self.to_tlwh()
        else:
            return self._static_tlwh

    def __str__(self):
        s = f'state={self.state} id={self.track_id:4d} age={self.age:5d} tsu={self.time_since_update} '
        s += f'tlbr={np.array(self.to_tlbr(), dtype=np.int)} static={self.static} stc={self.static_count},{self.static_unset_count} occ={self.partially_occluded}'
        return s
