#include <opencv2/opencv.hpp>
#include <opencv2/world.hpp>
using namespace cv;


std::vector<Rect> extract_cells(const Mat& src)
{
	Mat sudoku = src.clone();
	Mat grid = Mat(sudoku.size(), CV_8UC1);
	GaussianBlur(sudoku, sudoku, Size(11, 11), 0);

	adaptiveThreshold(sudoku, grid, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
	bitwise_not(grid, grid);

	Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));
	dilate(grid, grid, kernel);

	int count = 0;
	int max = -1;

	Point maxPt;

	for (int y = 0; y < grid.size().height; y++)
	{
		uchar* row = grid.ptr(y);
		for (int x = 0; x < grid.size().width; x++)
		{
			if (row[x] >= 128)
			{
				int area = floodFill(grid, Point(x, y), CV_RGB(0, 0, 64));

				if (area > max)
				{
					maxPt = Point(x, y);
					max = area;
				}
			}
		}
	}

	floodFill(grid, maxPt, CV_RGB(255, 255, 255));

	for (int y = 0; y < grid.size().height; y++)
	{
		uchar* row = grid.ptr(y);
		for (int x = 0; x < grid.size().width; x++)
		{
			if (row[x] == 64 && x != maxPt.x && y != maxPt.y)
			{
				int area = floodFill(grid, Point(x, y), CV_RGB(0, 0, 0));
			}
		}
	}

	erode(grid, grid, kernel);

	////////////////////

	Mat dst;
	Canny(grid, dst, 50, 200, 3);

	Mat kernel_d = getStructuringElement(MORPH_RECT, Size(25, 25));
	dilate(dst, dst, kernel_d);
	erode(dst, dst, kernel_d);

	imshow("dst", dst);

	std::vector<std::vector<Point>> cnts;
	findContours(dst, cnts, RETR_LIST, CHAIN_APPROX_SIMPLE);

	std::sort(cnts.begin(), cnts.end(), [](const std::vector<Point>& c1, const std::vector<Point>& c2) -> bool {
		return contourArea(c1) > contourArea(c2);
		});

	Rect rect;
	std::vector<Rect> ret;

	for (auto& c : cnts) {

		rect = boundingRect(c);
		ret.push_back(rect);
	}

	return ret;
}


std::vector<Point> order_points(const std::vector<Point>& pts)
{
	//pts has 4 elements
	Point tl, tr, bl, br;

	std::vector<Point> ptsTemp = pts;
	std::sort(ptsTemp.begin(), ptsTemp.end(), [](const Point& pt1, const Point& pt2)->bool {
		return pt1.x < pt2.x; //increasing
		});

	std::vector<Point> leftMost{ ptsTemp[0], ptsTemp[1] };
	std::vector<Point> rightMost{ ptsTemp[2], ptsTemp[3] };

	std::sort(leftMost.begin(), leftMost.end(), [](const Point& pt1, const Point& pt2)->bool {

		return pt1.y < pt2.y;
		});

	tl = leftMost[0];
	bl = leftMost[1];

	auto dist = [](const Point& p, const Point& q) {
		Point diff = p - q;
		return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
	};

	std::sort(rightMost.begin(), rightMost.end(), [&tl, dist](const Point& pt1, const Point& pt2)->bool {
		return dist(pt1, tl) < dist(pt2, tl);
		});

	tr = rightMost[0];
	br = rightMost[1];

	ptsTemp.clear();
	ptsTemp = { tl, tr, br, bl };
	return ptsTemp;
}

Mat four_point_transform(const Mat& image, std::vector<Point>& pts)
{
	std::vector<Point> vecRect = order_points(pts);
	Point2f rect[4];
	rect[0] = vecRect[0];
	rect[1] = vecRect[1];
	rect[2] = vecRect[2];
	rect[3] = vecRect[3];

	Point tl, tr, br, bl;
	tl = rect[0];
	tr = rect[1];
	br = rect[2];
	bl = rect[3];

	double widthA = sqrt(pow(br.x - bl.x, 2) + pow(br.y - bl.y, 2));
	double widthB = sqrt(pow(tr.x - tl.x, 2) + pow(tr.y - tl.y, 2));
	int maxWidth = max(int(widthA), int(widthB));

	double heightA = sqrt(pow(tr.x - br.x, 2) + pow(tr.y - br.y, 2));
	double heightB = sqrt(pow(tl.x - bl.x, 2) + pow(tl.y - bl.y, 2));
	int maxHeight = max(int(heightA), int(heightB));

	Point2f dst[4] = { Point2f(0,0), Point2f(maxWidth - 1, 0), Point2f(maxWidth - 1, maxHeight - 1), Point2f(0, maxHeight - 1) };
	Mat M = getPerspectiveTransform(rect, dst);
	Mat warped;

	warpPerspective(image, warped, M, Size(maxWidth, maxHeight));

	return warped;
}

Mat extract_digit(const Mat& cell)
{
	Mat thresh;
	threshold(cell, thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	std::vector<std::vector<Point>> cnts;
	findContours(thresh, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	if (cnts.size() == 0)
		return Mat::zeros(Size(0, 0), CV_8UC1);

	std::vector<Point> hull;
	std::vector < std::pair<std::vector<Point>, std::vector<Point>>> hull_contour;

	for (auto& c : cnts) {
		convexHull(c, hull);
		hull_contour.push_back(std::make_pair(hull, c));
	}

	auto comp = [](
		const std::pair<std::vector<Point>, std::vector<Point>>& p1,
		const std::pair<std::vector<Point>, std::vector<Point>>& p2) -> bool {
			return p1.first.size() < p2.first.size();
	};

	//sort by hull vector size
	std::sort(hull_contour.begin(), hull_contour.end(), comp);

	std::vector<Point> cnt = hull_contour.back().second;
	Rect border = boundingRect(cnt);
	Mat digit = thresh(border);

	return digit;
}

std::pair<Mat, Mat> find_puzzle(const Mat& image)
{
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat blurred;
	GaussianBlur(gray, blurred, Size(7, 7), 3);

	Mat thresh;
	adaptiveThreshold(blurred, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
	bitwise_not(thresh, thresh);

	std::vector<std::vector<Point>> cnts;
	findContours(thresh, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	sort(cnts.begin(), cnts.end(), [](const std::vector<Point>& a, const std::vector<Point>& b) -> bool {
		return contourArea(a) < contourArea(b);
		});

	Rect contourRect = boundingRect(cnts.back());

	std::vector<Point> approx;
	std::vector<Point> puzzleCnt(0);

	for (const auto& c : cnts)
	{
		double peri = arcLength(c, true);
		approxPolyDP(c, approx, 0.02 * peri, true);

		if (approx.size() == 4 && contourArea(c) > (0.4 * image.rows * image.cols))
		{
			puzzleCnt = approx;
			break;
		}
	}

	Mat puzzle = four_point_transform(image, puzzleCnt);
	Mat warped = four_point_transform(gray, puzzleCnt);

	waitKey(0);

	return { puzzle, warped };
}