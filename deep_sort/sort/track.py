# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    FullyOcculuded は画面から消えて検出結果がなくなった状態だが，トラッキングとしては
    継続状態にしたい状態のこと．_partially_occuluded = true かつ static = true の状態で，
    missed を実行されるとこの状態に入る．
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3
    FullyOcculuded = 4


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
        bbox の中心座標の変動が height * static_thresh 以下であり，それが static_set_frames 続いたら
        その track に static フラグを付ける．static_set_frames が 0 の場合，static フラグは付けない．
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, static_thresh=0.1, static_set_frames=0):
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
        self.partially_occuluded = False
        self.static_count = 0
        self.xyah_in_prev_frame = None

        self._n_init = n_init
        self._max_age = max_age
        self._static_thresh = _static_thresh
        self._static_set_frames = static_set_frames

    def to_xyah(self):
        '''Get current position in bounding box format (bbox center x, center y, aspect, height)
        '''
        return mean[:4].copy()

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
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

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
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        xyah = self.to_xyah()
        if self.xyah_in_prev_frame is not None:
            d = np.sum( (xyah[:2] - self.xyah_in_prev_frame[:2])**2 )
            if d < (xyah[3] * self.static_thresh)**2:
                self.static_count += 1
            else:
                self.static_count = 0

        self.xyah_in_prev_frame = xyah

        if self._static_set_frames > 0 and self.static_count > self._static_set_frames:
            self.static = True

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        elif self.state == FullyOcculuded:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.partially_occuluded and self.static and self.state == TrackState.Confirmed:
            self.state = TrackState.FullyOcculuded
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def mark_occuluded(self, flag=True):
        self.partially_occuluded = flag

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

    def is_fully_occuluded(self):
        return self.state == TrackState.FullyOcculuded

    def is_static_and_fully_occuluded(self):
        return self.is_static() and self.is_fully_occuluded()

    def is_partially_occuluded(self):
        return self.partially_occuluded and not self.is_fully_occuluded()
