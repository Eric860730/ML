import math


def countOutcome(outcome):
    success = 0
    fail = 0
    # The last element in outcome is '\n', skip it.
    for i in range(len(outcome) - 1):
        if (int(outcome[i]) == 1):
            success += 1
        else:
            fail += 1
    return success, fail


def binomialDistribution(a, b):
    proba = a / (a + b)
    return (proba**(a)) * ((1 - proba)**(b)) * math.factorial(a + b) / \
        (math.factorial(a) * math.factorial(b))


def onlineLearning(a, b, testfile):
    f = open(testfile, "r")
    outcome = f.readline()
    prior_a = a
    prior_b = b
    count = 1
    while outcome:
        success, fail = countOutcome(outcome)
        likelihood = binomialDistribution(success, fail)
        posterior_a = prior_a + success
        posterior_b = prior_b + fail
        print(f"case {count}: {outcome}", end="")
        print(f"Likelihood: {likelihood:.17f}")
        print(f"Beta prior:     a = {prior_a} b = {prior_b}")
        print(f"Beta posterior: a = {posterior_a} b = {posterior_b}\n")
        prior_a = posterior_a
        prior_b = posterior_b
        count += 1
        outcome = f.readline()
    f.close()
    exit()
