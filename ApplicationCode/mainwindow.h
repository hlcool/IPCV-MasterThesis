#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    QTimer* imageTimer;

private:
    Ui::MainWindow *ui;

public slots:
    void ProcessVideo();
private slots:
    void on_actionOpen_file_triggered();
};

#endif // MAINWINDOW_H