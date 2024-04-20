import cv2
import matplotlib.pyplot as plt


def convert_to_grayscale(image_path):
    """
    Converts an image to grayscale.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    img = cv2.imread(image_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale_img


def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        point1 (tuple): The coordinates of the first point.
        point2 (tuple): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def merge_dots(dots, radius):
    """
    Merges dots that are within a specified radius.

    Args:
        dots (list): The list of dots to be merged.
        radius (int): The radius within which dots should be merged.

    Returns:
        list: The merged dots.
    """
    merged_dots = []

    while dots:
        current_dot = dots.pop(0)
        merged_dots.append(current_dot)
        dots = [dot for dot in dots if euclidean_distance(
            current_dot, dot) > radius]

    return merged_dots


def calculate_smoothness(dot1, dot2, dot3):
    """
    Calculates the smoothness between three dots based on the angles between the vectors.

    Args:
        dot1 (tuple): The coordinates of the first dot.
        dot2 (tuple): The coordinates of the second dot.
        dot3 (tuple): The coordinates of the third dot.

    Returns:
        float: The smoothness score.
    """
    vector1 = (dot2[0] - dot1[0], dot2[1] - dot1[1])
    vector2 = (dot3[0] - dot2[0], dot3[1] - dot2[1])

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude_product = (vector1[0]**2 + vector1[1]
                         ** 2) * (vector2[0]**2 + vector2[1]**2)

    if magnitude_product == 0:
        return 0

    return abs(dot_product / magnitude_product)


def order_dots_smoothness_and_proximity(dots):
    """
    Orders the dots based on smoothness and proximity.

    Args:
        dots (list): The list of dots to be ordered.

    Returns:
        list: The ordered dots.
    """
    ordered_dots = [dots.pop(0)]

    while dots:
        current_dot = ordered_dots[-1]
        smoothness_scores = []

        for i in range(len(dots)):
            for j in range(i+1, len(dots)):
                dot1 = current_dot
                dot2 = dots[i]
                dot3 = dots[j]
                # Calculate the smoothness score based on the angles between the vectors
                smoothness = calculate_smoothness(dot1, dot2, dot3)
                # Also consider proximity
                proximity = euclidean_distance(current_dot, dot2)
                score = smoothness + (1 / proximity)
                smoothness_scores.append((score, i))

        if smoothness_scores:
            # Choose the dot with the highest combined score
            chosen_dot_index = max(smoothness_scores, key=lambda x: x[0])[1]
            chosen_dot = dots.pop(chosen_dot_index)
            ordered_dots.append(chosen_dot)
        else:
            break  # Break out of the loop if smoothness_scores is empty

    return ordered_dots


def find_dots(image, threshold=88, radius=5):
    """
    Finds dots in an image based on a threshold and radius.

    Args:
        image (numpy.ndarray): The grayscale image.
        threshold (int): The threshold value for pixel intensity.
        radius (int): The radius within which dots should be merged.

    Returns:
        list: The ordered dots.
    """
    # Find the coordinates of pixels darker than the threshold
    dots = [(x % image.shape[1], x // image.shape[1])
            for x, pixel_value in enumerate(image.flatten()) if pixel_value < threshold]

    
    # Merge dots within the specified radius
    merged_dots = merge_dots(dots, radius)

    # Order dots based on smoothness and proximity
    ordered_dots = order_dots_smoothness_and_proximity(merged_dots)

    return ordered_dots


def compute_cubic_spline_coefficients(x_coordinates, y_coordinates):
    """
    Computes the coefficients for cubic spline interpolation.

    Args:
        x (list): The x-coordinates of the dots.
        y (list): The y-coordinates of the dots.

    Returns:
        list: The coefficients for cubic spline interpolation.
    """
    n = len(x_coordinates)
    h = [x_coordinates[i] - x_coordinates[i - 1] for i in range(1, n)]
    alpha = [(3 / h[i]) * (y_coordinates[i + 1] - y_coordinates[i]) - (3 / h[i - 1])
             * (y_coordinates[i] - y_coordinates[i - 1]) for i in range(1, n - 1)]

    # Initialize lists with zeros
    l, mu, z = [0] * n, [0] * n, [0] * n
    l[0] = 1
    mu[0] = 0
    z[0] = 0

    for i in range(1, n - 1):
        l[i] = 2 * (x_coordinates[i + 1] - x_coordinates[i - 1]) - \
            h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    l[-1] = 1
    z[-1] = 0
    c, b, d = [0] * n, [0] * (n - 1), [0] * (n - 1)
    c[-1] = 0

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y_coordinates[j + 1] - y_coordinates[j]) / \
            h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    coefficients = []
    for i in range(n - 1):
        coefficients.append((y_coordinates[i], b[i], c[i], d[i]))
    return coefficients


def evaluate_cubic_spline(coefficients, x, x_i):
    """
    Evaluates the cubic spline at a given x-coordinate.

    Args:
        coefficients (list): The coefficients for cubic spline interpolation.
        x (list): The x-coordinates of the dots.
        x_i (float): The x-coordinate at which to evaluate the cubic spline.

    Returns:
        float: The y-coordinate of the cubic spline at the given x-coordinate.
    """
    n = len(coefficients)
    for i in range(n):
        if x_i >= x[i] and x_i <= x[i + 1]:
            a, b, c, d = coefficients[i]
            dx = x_i - x[i]
            return a + b * dx + c * dx ** 2 + d * dx ** 3


def cubic_spline_interpolate(coefficients, x, indices):
    """
    Interpolates a cubic spline at given x-coordinates.

    Args:
        coefficients (list): The coefficients for cubic spline interpolation.
        x (list): The x-coordinates at which to interpolate.
        indices (list): The indices of the dots.    

    Returns:
        list: The interpolated values.
    """

    interpolated_values = [evaluate_cubic_spline(
        coefficients, indices, xi) for xi in x]
    return interpolated_values


def dot_product(v1, v2):
    """
    Calculates the dot product of two vectors.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    return sum(x * y for x, y in zip(v1, v2))


def transpose(matrix):
    """
    Transposes a matrix.

    Args:
        matrix (list): The matrix to be transposed.

    Returns:
        list: The transposed matrix.
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def matrix_multiply(matrix1, matrix2):
    """
    Multiplies two matrices.

    Args:
        matrix1 (list): The first matrix.
        matrix2 (list): The second matrix.

    Returns:
        list: The product of the two matrices.
    """
    return [[dot_product(row, col) for col in transpose(matrix2)] for row in matrix1]


def solve_system(matrix_A, vector_b):
    """
    Solves a system of linear equations.

    Args:
        matrix_A (list): The coefficient matrix.
        vector_b (list): The constant vector.

    Returns:
        list: The solution vector.
    """
    n = len(vector_b)
    coefficients = [0] * n

    # Forward elimination
    for i in range(n):
        pivot = matrix_A[i][i]
        for j in range(i + 1, n):
            factor = matrix_A[j][i] / pivot
            for k in range(n):
                matrix_A[j][k] -= factor * matrix_A[i][k]
            vector_b[j] -= factor * vector_b[i]

    # Backward substitution
    for i in range(n - 1, -1, -1):
        coefficients[i] = vector_b[i] / matrix_A[i][i]
        for j in range(i - 1, -1, -1):
            vector_b[j] -= matrix_A[j][i] * coefficients[i]

    return coefficients


def least_squares_curve_fit(x, y, degree):
    """
    Performs least squares curve fitting.

    Args:
        x (list): The x-coordinates of the dots.
        y (list): The y-coordinates of the dots.
        degree (int): The degree of the polynomial to fit.

    Returns:
        list: The coefficients of the polynomial.
    """
    n = len(x)
    A = [[x[i] ** j for j in range(degree + 1)] for i in range(n)]
    ATA = matrix_multiply(transpose(A), A)
    ATy = matrix_multiply(transpose(A), [[yi] for yi in y])
    coefficients = solve_system(ATA, [row[0] for row in ATy])
    return coefficients


def evaluate_polynomial(coefficients, x):
    """
    Evaluates a polynomial at given x-coordinates.

    Args:
        coefficients (list): The coefficients of the polynomial.
        x (list): The x-coordinates at which to evaluate the polynomial.

    Returns:
        list: The y-coordinates of the polynomial at the given x-coordinates.
    """
    result = [0] * len(x)
    for i in range(len(coefficients)):
        result = [result[j] + coefficients[i] *
                  (xi ** i) for j, xi in enumerate(x)]
    return result


def plot_input_image(ax, grayscale_image):
    ax.imshow(grayscale_image, cmap='gray', aspect='auto')
    ax.set_title('Input Image')


def plot_interpolation(ax, ordered_dots, interpolation_color='red'):
    ax.scatter(*zip(*ordered_dots), s=1, color='black')
    ax.set_title('Interpolation')
    ax.plot(*zip(*ordered_dots+[ordered_dots[0]]),
            color=interpolation_color, linewidth=1)


def plot_cubic_spline(ax, ordered_dots):
    ax.scatter(*zip(*ordered_dots), s=1, color='black')
    ax.set_title('Cubic Spline')

    x_values = [dot[0] for dot in ordered_dots] + [ordered_dots[0][0]]
    y_values = [dot[1] for dot in ordered_dots] + [ordered_dots[0][1]]

    indices = list(range(len(x_values)))
    num_points = 1000

    x = [i * (len(x_values) - 1) / (num_points - 1) for i in range(num_points)]
    y = [i * (len(y_values) - 1) / (num_points - 1) for i in range(num_points)]

    # Compute cubic spline coefficients
    x_coefficients = compute_cubic_spline_coefficients(indices, x_values)
    y_coefficients = compute_cubic_spline_coefficients(indices, y_values)

    # Interpolate x and y values
    x_interpolated = cubic_spline_interpolate(x_coefficients, x, indices)
    y_interpolated = cubic_spline_interpolate(y_coefficients, y, indices)

    # Plot the interpolated curve
    ax.plot(x_interpolated, y_interpolated, color='blue', linewidth=1)


def plot_least_squares(ax, ordered_dots, power):
    ax.scatter(*zip(*ordered_dots), s=1, color='black')
    ax.set_title('Least Square')

    x_values = [dot[0] for dot in ordered_dots] + [ordered_dots[0][0]]
    y_values = [dot[1] for dot in ordered_dots] + [ordered_dots[0][1]]

    # Get the indices
    indices = list(range(len(x_values)))
    num_points = 1000

    x = [i * (len(x_values) - 1) / (num_points - 1) for i in range(num_points)]
    y = [i * (len(y_values) - 1) / (num_points - 1) for i in range(num_points)]

    # Perform least squares curve fitting
    coefficients_x = least_squares_curve_fit(indices, x_values, power)
    coefficients_y = least_squares_curve_fit(indices, y_values, power)

    # Interpolate x and y values
    x_interpolated = evaluate_polynomial(coefficients_x, x)
    y_interpolated = evaluate_polynomial(coefficients_y, y)

    # Plot the interpolated curve
    ax.plot(x_interpolated, y_interpolated, color='green', linewidth=1)


def run(path, radius=100, threshold=88, power=6):
    # Load the image
    image_path = path
    grayscale_image = convert_to_grayscale(image_path)

    _, axs = plt.subplots(2, 2, figsize=(10, 10))

    plot_input_image(axs[0, 0], grayscale_image)

    # Plot dots for pixels darker than the threshold within the specified radius
    # Task 1
    ordered_dots = find_dots(grayscale_image, threshold=threshold,
                             radius=radius)
    # Task 2
    plot_interpolation(axs[0, 1], ordered_dots)

    # Task 3
    plot_cubic_spline(axs[1, 0], ordered_dots)
    # Task 4
    plot_least_squares(axs[1, 1], ordered_dots, power=power)

    # Invert y-axis to match image orientation
    for ax in axs.flatten()[1:]:
        ax.invert_yaxis()

    # for i, dot in enumerate(ordered_dots):
    #     axs[0, 1].text(dot[0], dot[1], str(i), color='red', fontsize=8)

    plt.tight_layout()
    # Task 5, actually it is all the functions and plotting above
    plt.show()
    # Task 6 in separate file


o_radius4_overlapping_curve = 30
o_radius4_close_edged_curve = 70
o_radius4_n_overlapping_curve = 100
run(path="C:/Users/nkhal/Desktop/NP/Week11/dd.jpg",
    radius=30, threshold=88, power=20)
