#include "widget.h"
#include <QtWidgets/QFileDialog>
#include "mmod.h"
#include "ui_widget.h"

Widget::Widget(QWidget *parent) : QWidget(parent), ui(new Ui::Widget) {
  ui->setupUi(this);
}

Widget::~Widget() { delete ui; }

void Widget::setObjectPixmap(QPixmap pixmap) {
  ui->objectPixmap->setPixmap(pixmap);
}

void Widget::setMMOD(Annotator::Plugins::MMOD *mmod) { this->mmod = mmod; }

void Widget::setProgress(int percent) { ui->progressBar->setValue(percent); }

void Widget::on_trainButton_clicked() {
  if (training) {
    training = false;
    ui->trainButton->setText(tr("Train"));
    this->mmod->stop();
  } else {
    ui->trainButton->setText(tr("Stop Training"));
    training = true;
    this->mmod->train();
  }
}

void Widget::on_saveButton_clicked() {
  QString fileName = QFileDialog::getSaveFileName(
      this, tr("Save Trained File"), "", tr("dnn Net Data (*.dat)"));
  mmod->saveNet(fileName.toStdString());
}

void Widget::on_loadButton_clicked() {
  QString fileName = QFileDialog::getOpenFileName(
      this, tr("Load Trained File"), "", tr("dnn Net Data (*.dat)"));
  mmod->loadNet(fileName.toStdString());
}
