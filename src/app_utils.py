
def get_guess_loss(guess, fake, boost = 10) -> float:
    if fake:
        return (100 - guess) * boost
    else:
        return guess * boost