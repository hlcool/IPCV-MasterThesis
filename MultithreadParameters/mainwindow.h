#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cameraworker.h"
#include "camerastream.h"

#define MAX_NUM_CAM 3

using namespace cv;

namespace Ui {
class MainWindow;
}

class QThread;
class QLabel;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(String ParamPDDetectorCons, double ParamPDThresholdCons, bool ParamSemanticFitleringCons, bool ParamMultiCameraCons, QWidget *parent = 0);
    ~MainWindow();

    void threadStarting();
    void connectSignals2Slots(QThread *thread, CameraWorker *worker);
    void fillCameraVariables(CameraStream &Camera, int i);

    // Camera number of sampled views
    int NViews = 9;

    String ParamPDDetector;
    double ParamPDThreshold;
    bool ParamSemanticFitlering, ParamMultiCamera;

signals:

private slots:
    void sharePedestrianDetections(int CameraNumber);

    void on_actionOpen_file_triggered();
    void on_actionCamera_1_triggered();
    void on_actionCamera_2_triggered();
    void on_actionCamera_3_triggered();

    void on_actionAbout_triggered();

private:
    Ui::MainWindow *ui;
    int numCams;
    QLabel *labels[MAX_NUM_CAM];
    QThread* threads[MAX_NUM_CAM];
    CameraWorker* CameraWorkers[MAX_NUM_CAM];
    String VideoPaths[MAX_NUM_CAM];
    QStringList filenames;
    QString GlobalPath;
};

#endif // MAINWINDOW_H
