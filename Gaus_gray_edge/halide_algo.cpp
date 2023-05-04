#include <Halide.h>
#include <opencv2/opencv.hpp>

using namespace Halide;

class EdgeDetection : public Halide::Generator<EdgeDetection> {
public:
    int t_channels;
    EdgeDetection(int e_channels): t_channels(e_channels){}
    Input<Buffer<uint8_t>> input{"input", 3};

    Output<Buffer<uint8_t>> output{"output", 3};

    void generate() {
        // Define the variables and helper functions we will use
        Var x("x"), y("y"), c("c");

        // Convert the input to grayscale
        Func gray("gray");
        gray(x, y) = cast<float>(0.299f * input(x, y, 0) + 0.587f * input(x, y, 1) + 0.114f * input(x, y, 2));

        // Apply a Gaussian blur to the grayscale image
        float sigma = 1.5f;
        int radius = std::ceil(3.0f * sigma);
        Func blur_x("blur_x");
        blur_x(x, y) = (1.0f / (2.0f * 3.14f * sigma * sigma)) * exp(-((x - radius) * (x - radius) + (y - radius) * (y - radius)) / (2.0f * sigma * sigma)) * gray(x - radius, y);
        Func blur_y("blur_y");
        blur_y(x, y) = (1.0f / (2.0f * 3.14f * sigma * sigma)) * exp(-((x - radius) * (x - radius) + (y - radius) * (y - radius)) / (2.0f * sigma * sigma)) * blur_x(x, y - radius);

        // Compute the Sobel gradients
        Func gradient_x("gradient_x");
        gradient_x(x, y) = -blur_y(x - 1, y - 1) + blur_y(x + 1, y - 1) - 2.0f * blur_y(x - 1, y) + 2.0f * blur_y(x + 1, y) - blur_y(x - 1, y + 1) + blur_y(x + 1, y + 1);
        Func gradient_y("gradient_y");
        gradient_y(x, y) = -blur_y(x - 1, y - 1) - 2.0f * blur_y(x, y - 1) - blur_y(x + 1, y - 1) + blur_y(x - 1, y + 1) + 2.0f * blur_y(x, y + 1) + blur_y(x + 1, y + 1);

        // Compute the magnitude and direction of the gradients
        Func magnitude("magnitude");
        magnitude(x, y) = sqrt(gradient_x(x, y) * gradient_x(x, y) + gradient_y(x, y) * gradient_y(x, y));
        Func direction("direction");
        direction(x, y) = atan2(gradient_y(x, y), gradient_x(x, y));

        // Compute the thresholded edge map
        float threshold = 50.0f;
        Func edge_map("edge_map");
        edge_map(x, y) = select(magnitude(x, y) > threshold, cast<uint8_t>(255), cast<uint8_t>(0));

                // Convert the output back to 3 channels
        Func output_3c("output_3c");
        output_3c(x, y, c) = cast<uint8_t>(edge_map(x, y)) * select(c == 0, 1, select(c == 1, 1, 1));

        output(x, y, c) = output_3c(x, y, c);

        Func f{"f"};
        f(x,y,c)=input(x,y,c);
        // Schedule the pipeline
        int width = 1440;
        int height = 1080;
        int channels = t_channels;

        // Schedule the pipeline
        output.bound(c, 0, channels)
              .unroll(c)
              .reorder(c, x, y)
              .bound(x, radius, width - radius)
              .bound(y, radius, height - radius)
              .unroll(x, 8)
              .unroll(y, 8);

        output_3c.compute_at(output, x).vectorize(x, 8);

        gray.compute_root();
        blur_y.compute_at(output_3c, x).vectorize(x, 8);
        gradient_x.compute_at(output_3c, x).vectorize(x, 8);
        gradient_y.compute_at(output_3c, x).vectorize(x, 8);
        magnitude.compute_at(output_3c, x).vectorize(x, 8);
        direction.compute_at(output_3c, x).vectorize(x, 8);
        edge_map.compute_at(output_3c, x).vectorize(x, 8);
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: ./edge_detection input_image_path\n");
        return 0;
    }

    // Load the input image
    cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat output_image;

    // Convert the input to a Halide buffer
    Buffer<uint8_t> input(input_image.data, input_image.cols, input_image.rows, input_image.channels());
    Buffer<uint8_t> output(output_image);

    // Perform edge detection
    EdgeDetection edge_detection(input_image.channels());
    Buffer<uint8_t> output = edge_detection.process(input);

    // Convert the output to a OpenCV Mat and display the result
    cv::Mat output_image(output.height(), output.width(), CV_8UC(output.channels()), output.data());
    cv::imshow("Edge Detection", output_image);
    cv::waitKey();

    return 0;
}
