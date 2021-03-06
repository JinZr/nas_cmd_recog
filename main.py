from policy_gradient import PolicyGradient

def main():

    policy_gradient_ins = PolicyGradient(
        train_set=None, 
        test_set=None
    )
    policy_gradient_ins.solve_environment()


if __name__ == '__main__':
    main()
