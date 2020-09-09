#include <opencv2/opencv.hpp>
#include <opencv2/world.hpp>
#include <utility>
#include "digitrecognizer.h"


using namespace cv;

Mat four_point_transform(const Mat& image, std::vector<Point>& pts);
Mat extract_digit(const Mat& cell);
std::vector<Point> order_points(const std::vector<Point>& pts);
std::pair<Mat, Mat> find_puzzle(const Mat& image);
std::vector<Rect> extract_cells(const Mat& src);


int main()
{
	DigitRecognizer* dr = new DigitRecognizer();
	dr->loadDigits();
	dr->prepareDigits();

	Mat image;
	Mat cell;
	Mat digit;

	//processing
	std::string filename = "sudoku2.jpg";
	image = imread(filename, IMREAD_UNCHANGED);

	Mat puzzleImage, warped, warped_rgb;
	std::pair<Mat, Mat> pair = find_puzzle(image);

	puzzleImage = pair.first;
	warped = pair.second;
	imshow("warped", warped);


	resize(puzzleImage, puzzleImage, Size(1000, 1000));
	resize(warped, warped, Size(1000, 1000));

	cvtColor(warped, warped_rgb, COLOR_GRAY2BGR);

	std::vector<int> numbers;

	//find grid
	std::vector<Rect> rect_cells = extract_cells(warped);

	std::sort(rect_cells.begin(), rect_cells.end(), [](const Rect& r1, const Rect& r2) {
		bool c = r1.tl().x < r2.tl().x;
		if (c == 0)
			c = r1.tl().y < r1.tl().y;
		return c;
		});

	std::cout << rect_cells.size() << std::endl;

	if (rect_cells.size() == 82) { 
		for (size_t i = 1; i < rect_cells.size(); i++)
		{
			
			cell = warped(rect_cells[i]);
			digit = extract_digit(cell);

			if (digit.size() != Size(0, 0)) {

				int number = dr->classify(digit);
				numbers.push_back(number);
				putText(warped_rgb, std::to_string(number), Point(rect_cells[i].x, rect_cells[i].y + rect_cells[i].height), FONT_HERSHEY_COMPLEX,2, Scalar(0, 0, 255));
			}
		}
	}
	else {

		float stepX = warped.size().height / 9;
		float stepY = warped.size().width / 9;

		std::vector<Vec4f> row;
		for (size_t y = 0; y < 9; y++)
		{
			for (size_t x = 0; x < 9; x++)
			{
				float startX = x * stepX;
				float startY = y * stepY;

				float endX = (x + 1) * stepX;
				float endY = (y + 1) * stepY;

				row.emplace_back(Vec4f(startX, startY, endX, endY));

				Rect2f rect(Point2f(startX, startY), Point2d(endX, endY));
				/*rect.x += 15;
				rect.y += 15;
				rect.width -= 15;
				rect.height -= 15;*/

				//bound rect
				if (rect.x + rect.width > warped.cols)
					continue;
				if (rect.y + rect.height > warped.rows)
					continue;

				cell = warped(rect);
				digit = extract_digit(cell);

				if (digit.size() != Size(0,0)) {

					int number = dr->classify(digit); 
					putText(warped_rgb, std::to_string(number), Point(startX + 5, endY - 5), FONT_HERSHEY_COMPLEX,2, Scalar(0, 0, 255));
					
				}
			}
		 }
	}
	waitKey();

	return 0;
}
