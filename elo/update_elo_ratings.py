"""
This script takes a dataframe of match outcomes, assumed to have a column called 'winner'
that takes one of two values: 1 or 2.

We update initial Elo ratings based on this dataframe of matches.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse


def predict(s1, s2, beta):
    """
    Predict the outcome of a match between players 1 and 2, given their rating scores
    s1, s2.

    Args:
    s1: float; rating score of player 1.
    s2: float; rating score of player 2.

    Returns:
    int; either 1 or 2, representing the player that will most likely win.
    """
    ratio = (s1 - s2) / (beta * np.sqrt(2))
    proba = stats.norm.cdf(ratio)
    return proba
    

def update(s1, s2, winner, alpha, beta):
    """
    Return updated S1, S2 given data with competition winners column.

    Args:
    s1: float; rating score of player 1.
    s2: float; rating score of player 2.
    res: int; either 1 or 2. Indicates winner of a match.
    alpha: float;  0 < alpha < 1. The learning rate.
    beta: float; the standard deviation of the normal distribution that we use.

    Returns:
    (s1, s2): Tuple[float]; updated values for s1 and s2.
    """
    # define K, Y, and Delta
    K = alpha * beta * np.sqrt(np.pi)
    if winner == 1:
        Y = 1.
    elif winner == 2:
        Y = -1.
    else:
        Y = 0.
    ratio = (s1 - s2) / (beta * np.sqrt(2))
    Delta = K * ( ((Y+1.) / 2.) - stats.norm.cdf(ratio) )
    return (s1+Y*Delta, s2-Y*Delta)
    
    
# ===== ===== ===== ===== =====
if __name__ == "__main__":
    # cli interface
    parser = argparse.ArgumentParser(description="Update the Elo ratings for two players given data.")
    parser.add_argument("--data", required=True, help="Path to a CSV file containing column 'winner'.")
    parser.add_argument("--alpha", required=False, default=0.5,
                        help="The learning rate. Should satisfy 0 < alpha < 1.")
    parser.add_argument("--beta", required=False, default=16.817,
                        help="The stdv of the scores' normal distribution. Must be positive.")
    args = parser.parse_args()

    # input values & validation
    df = pd.read_csv(args.data)
    assert ('winner' in df.columns)
    assert len(np.unique(df.winner.values)) == 2
    s1, s2 = 1.0, 1.0

    # main function
    alpha = float(args.alpha)
    beta = float(args.beta)
    print(f"[Initial Values] | s1={s1} | s2={s2} | Alpha={alpha} | Beta={beta}")
    for k,winner in enumerate(df.winner.values):
        s1, s2 = update(s1, s2, winner, alpha, beta)
        print(f"[Iter: {k}] | s1={s1} | s2={s2}")
