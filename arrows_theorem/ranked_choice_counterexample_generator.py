"""
Finds counterexamples to any of arrow's theorem's conditions for N voters in a ranked-choice ballot
across three candidates.
"""
import numpy as np
from itertools import permutations
from tqdm import trange, tqdm

INT = np.int32

def sample_preferences(nvoters, ncands=3):
    """
    * nvoters: int; number of voters to sample.
    * ncands: int (default 3); number of candidates. Keep this low because of
      combinatorial blowup. Suggest <= 5, as 5! == 120.

    Returns:
      N x K array where arr[i] = ranking of the K elements with
        arr[i][0] = bottom-ranked for voter i,
        ...
        arr[i][K-1] = top-ranked for voter i.
    """
    # generate K! x K array of possible candidate rankings:
    # (N.B.: candidates are labelled 0, 1, 2, ..., (K-1).)
    candidates = list(range(ncands))
    possible_profiles = np.array(list(permutations(candidates)), dtype=INT)
    # generate N samples from possible profiles and return:
    sampled_indices = np.random.choice(len(possible_profiles), size=nvoters, replace=True)
    sampled_prefs = possible_profiles[sampled_indices]
    return sampled_prefs


def compute_social_ranking(prefs):
    """
    Given a set of preferences, return the social ranking as an array of length K
    (where K is the size of axis 1 of `prefs ~ N x K`).

    Uses the following rules:
    (1) All first-choice votes are counted. If a candidate receives more than 50% of
        first-choice votes, that candidate wins.
    (2) If no candidate earns more than 50% of first-choice votes, then counting will continue in rounds.
    (3) At the end of each round, the last-place candidate is eliminated and voters who chose that
        candidate now have their vote counted for their next choice.
    (4) Your vote is counted for your second choice only if your first choice is eliminated.
        If both your first and second choices are eliminated, your vote is counted for your next
        choice, and so on.
    (5) This process continues until there are two candidates left. The candidate with the most votes wins.
    """
    # K x K matrix of vote counts: construct counts ~ K x K, where counts[i][c] indicates the number of
    # votes for candidate c in the i-th position.
    # (N.B.: `np.unique` always returns values in sorted order, so no need to save the unique values.)
    N, K = prefs.shape
    counts = np.zeros((K,K), dtype=INT)
    for idx in range(K):
        _, counts[idx,:] = np.unique(prefs[:,idx], axis=0, return_counts=True)

    # --- check if a candidate wins >50% of first choice votes:
    if np.max(counts[-1,:]) > N // 2:
        cands = np.arange(prefs.shape[-1], dtype=INT)
        # return top votes as the social ranking
        sorted_ranking = np.array(sorted(zip(cands, counts[-1,:]), key=lambda xy: xy[1]))
        return sorted_ranking[:,0]

    # --- perform elimination, round-by-round:
    ranking = []
    vote_counter = { k: 0 for k in range(K) }
    # TODO: finish
    
    return social_ranking


def check_unanimity(prefs, social_rank):
    """
    Check the unanimity condition.
    
    * prefs: N x K matrix representing 
    * social_rank: (...)

    Returns: boolean; True if satisfies unanimity condition, False otherwise.
    """
    return True # TODO


def check_rationality(prefs, social_rank):
    """
    Check the rationality condition.
    """
    return True # TODO


def check_iia(prefs, social_rank):
    """
    Check the IIA ("independence of irrelevant alternatives") condition.
    """
    return True # TODO


if __name__ == '__main__':
    # --- parse and validate arguments:
    parser = argparse.ArgumentParser(description="Search for situations where ranked choice fails one of the four arrow conditions.")
    parser.add_argument("--ncand", default=3, help="Number of candidates.")
    parser.add_argument("--nvoter", default=10, help="Number of voters.")
    parser.add_arguments("--nsamples", default=1000, help="Number of times to sample and check conditions.")
    args = parser.parse_args()
    assert args.nsamples > 1
    assert args.ncand >= 3
    assert args.nvoter >= 3

    # --- monte carlo loop:
    for _ in trange(args.nsamples):
        preferences = sample_preferences(args.nvoter, args.ncand)
        social_rank = compute_social_ranking(preferences)
        iia_bool = check_iia(preferences, social_rank)
        rationality_bool = check_rationality(preferences, social_rank)
        unanimity_bool = check_unanimity(preferences, social_rank)
        conditions = all([ iia_bool, rationality_bool, unanimity_bool ])
        if not conditions:
            print("Failing ranked choice found!")
            print("Preferences profile:")
            print(preferences)
            print("Computed social rank:")
            print(social_rank)
            print("Conditions:")
            print(f"> iia_bool={iia_bool}")
            print(f"> rationality_bool={rationality_bool}")
            print(f"> unanimity_bool={unanimity_bool}")
            print("Terminating the sample loop early.")
            break

    # --- no breaking example found: print message and quit
    print("No breaking example found in {args.nsamples} samples. Try different `--nvoter` and/or `--ncand`.")
