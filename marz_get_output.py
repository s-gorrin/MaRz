"""
Attribution:
These functions were initially written for the iris dataset by Dr. Eric Braude and then later generalized.
"""


VAL, WT = 0, 1  # labels for convenience
SMALL_DELTA = 0.0001


def add_output_contributions(a_contribution, an_input, a_datum, a_fuzzy_width):  # see [1]
    """
    INTENT: Add the fuzzy contribution of a_datum to some_contributions relative to an_input

    PRECONDITION 1 (a_contribution) = [value, weight_] reals, weight_ >= 0
    PRE2 (an_input) = a list of reals
    PRE3 (a_fuzzy_width) = non-negative floats <=1 = half width of fuzzy triangle per input field
    PRE4 (Nonzero contributions): |an_input[i] - a_datum[i]| < a_fuzzy_width[i]
        for i < len(an_input)

    POST-CONDITION 1 (weight_) = minimum fuzzy value from features of an_input,
        from the corresponding triangles centered at a_datum[s]
        with width 2 * a_fuzzy_width[s]--for s = 0, 1, ...--

    POST2 (Contribution to Output): output_weight = weight_ * (2 - weight_) AND
        a_contribution[VAL] = old(a_contribution[VAL]) + (output_weight * a_datum[-1]) and
        a_contribution[WT] = old(a_contribution[WT]) + output_weight
    """

    # --- [O1] (fuzzy_slope) = slope of the left triangle side
    fuzzy_slope = [1 / w for w in a_fuzzy_width]

    # --- [O2] = POST1 (weight_)
    weight_ = 1.0  # seeking min, so initialize to max
    for s in range(len(an_input)):
        horizontal_distance = abs(an_input[s] - a_datum[s])
        # slope = rise / run  = weight_ / horizontal_distance--for non-zero run--so ...
        temp_weight = fuzzy_slope[s] * horizontal_distance if horizontal_distance != 0 else 1.0
        if temp_weight < weight_:
            weight_ = temp_weight  # select minimum weight_

    # --- [O3] = POST2 (Contribution to Output)
    output_weight = weight_ * (2 - weight_)
    a_contribution[VAL] += (output_weight * a_datum[-1])
    a_contribution[WT] += output_weight


def get_output(an_input, some_data, a_fuzzy_width, indices_in_width):
    """
    NUM_INPUTS = the number of features in the data

    PRECONDITION 1 (an_input) = NUM_INPUTS positive reals
    PRE2 (some_data) = a non-empty list of lists of NUM_INPUTS positive reals ordered left-to-right
        with targets at column [-1]
    PRE3 (a_fuzzy_width) = NUM_INPUTS non-negative floats <=1 = half width fuzzy triangle per field

    POST-CONDITION: --as for add_output_contributions(contribution_) for every a_datum
    in the (hyper-)rectangle defined by min_fuzzy and max_fuzzy, exclusive,
    where contribution_ is initially [0, SMALL_DELTA].

    RETURNS contribution_[VAL] / contribution_[WT]
    """

    contribution = [0, SMALL_DELTA]

    for index in indices_in_width:
        add_output_contributions(contribution, an_input, some_data[index], a_fuzzy_width)

    return contribution[VAL] / contribution[WT]
