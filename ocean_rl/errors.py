"""Training failure types shared by experiment runners."""
class EpisodeTruncatedError(RuntimeError):
    """An incomplete episode cannot be used by this episodic learner."""
