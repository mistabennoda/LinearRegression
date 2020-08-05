from numpy import *


def compute_error_for_line_given_point(b, m, points):
    totalError = 0
    # for every ponit
    for i in range(0, len(points)):
        # get the xy value
        x = points[i, 0]
        y = points[i, 1]
        # get the differences, add it and get it to the total
        temp1 = m*x
        temp2 = temp1 + b
        temp3 = y - temp2
        temp4 = temp3 ** 2
        totalError += temp4
        '''totalError += (y-(m*x + b)) ** 2'''

    # get the average
    return totalError / float(len(points))


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    # gradient descent
    for i in range(num_iterations):
        # update b and m with the new more accurate b and m by performing this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]


def step_gradient(b_current, m_current, points, learningRate):
    # starting points for the gradients
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respect to b and m
        # computing partial derivatives of the error function
        b_gradient += -(2/N) * (y-(m_current * x + b_current))
        m_gradient += (2/N) * x * (y-(m_current * x + b_current))

    # updating b and m values using the partial derivative
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


# target: calculate the slope
def run():
    # Step1: collect data
    # xy value pairs
    # x - amount of hours studied
    # y - test score
    points = genfromtxt('data.csv', delimiter=',')

    # Step2: define hyper-parameters
    # how fast should our model converge
    learning_rate = 0.0001
    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 500

    # Step3: fit, train the model
    print('starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error_for_line_given_point(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print('ending gradient descent at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_point(b, m, points)))


if __name__ == '__main__':
    run()
