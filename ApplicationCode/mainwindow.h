#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <string>
#include <fstream>
#include <iostream>

using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    // Flag used to enable the QFile dialog or just select always the same video
    // 0 - Automatically selects the video from the path
    // 1 - Enables the QFileDialog
    int ProgramFlag = 0;

    // CVWidget size
    int WidgetHeight, WidgetWidth;

    void DisplayImages(string FrameNumber);

private:
    Ui::MainWindow *ui;
    QTimer* imageTimer;

public slots:
    void ProcessVideo();
private slots:
    void on_actionOpen_file_triggered();
    void on_actionCamera_1_triggered();
    void on_actionCamera_2_triggered();
};

#endif // MAINWINDOW_H
