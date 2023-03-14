"""
This program implements functions to test the code that creates our
plots and that calculates accuracy, precision, recall, and F-scores.
"""


def test_scores() -> None:
    """
    Calculates the accuracy, precision, recall, and F-scores by hand.
    """
    accuracy = (0 + 364 + 26) / 585
    p_app = 0
    p_indiff = 364 / (34 + 364 + 146)
    p_ran = 26 / (2 + 13 + 26)
    precision = (p_app + p_indiff + p_ran) / 3
    r_app = 0
    r_indiff = 364 / (0 + 364 + 13)
    r_ran = 26 / (0 + 146 + 26)
    recall = (r_app + r_indiff + r_ran) / 3
    f_app = 0
    f_indiff = 2 * ((p_indiff * r_indiff) / (p_indiff + r_indiff))
    f_ran = 2 * ((p_ran * r_ran) / (p_ran + r_ran))
    f_score = (f_app + f_indiff + f_ran) / 3
    print('Accuracy score:', accuracy)
    print('Precision score:', precision)
    print('Recall score:', recall)
    print('F-score:', f_score)


def main():
    test_scores()


if __name__ == '__main__':
    main()
