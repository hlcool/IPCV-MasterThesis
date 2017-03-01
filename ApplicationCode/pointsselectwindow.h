#ifndef POINTSSELECTWINDOW_H
#define POINTSSELECTWINDOW_H

#include <QDialog>

using namespace std;

namespace Ui {
class pointsselectwindow;
}

class pointsselectwindow : public QDialog
{
    Q_OBJECT

public:
    explicit pointsselectwindow(QWidget *parent = 0);
    ~pointsselectwindow();

    // CVWidget size
    int PointsWidgetHeight, PointsWidgetWidth;
    vector<int> x, y;

private:
    Ui::pointsselectwindow *ui;
};

#endif // POINTSSELECTWINDOW_H
