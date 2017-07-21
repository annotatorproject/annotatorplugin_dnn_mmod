
// source: http://dlib.net/dnn_mmod_ex.cpp.html
#include "mmod.h"
#include "widget.h"

#include <AnnotatorLib/Annotation.h>
#include <AnnotatorLib/Commands/NewAnnotation.h>
#include <AnnotatorLib/Frame.h>
#include <AnnotatorLib/Session.h>

#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/svm_threaded.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ctype.h>
#include <iostream>
#include <memory>

#include <chrono>
#include <thread>

using namespace Annotator::Plugins;

Annotator::Plugins::MMOD::MMOD() { widget.setMMOD(this); }

MMOD::~MMOD() {}

QString MMOD::getName() { return "MMOD"; }

QWidget *MMOD::getWidget() { return &widget; }

bool MMOD::setFrame(shared_ptr<Frame> frame, cv::Mat image) {
  this->lastFrame = this->frame;
  this->frame = frame;
  this->frameImg = image;
  return lastFrame != frame;
}

// first call
void MMOD::setObject(shared_ptr<Object> object) {
  if (object != this->object) {
    this->object = object;
    widget.setObjectPixmap(getImgCrop(object->getFirstAnnotation(), 96));
  }
}

shared_ptr<Object> MMOD::getObject() const { return object; }

void MMOD::setLastAnnotation(shared_ptr<Annotation> /*annotation*/) {}

std::vector<shared_ptr<Commands::Command>> MMOD::getCommands() {
  std::vector<shared_ptr<Commands::Command>> commands;
  if (object == nullptr || frame == nullptr || lastFrame == nullptr ||
      lastFrame == frame)
    return commands;

  try {
    cv::Rect res = findObject();

    if (res.width > 0 && res.height > 0) {
      int x = res.x;
      int y = res.y;
      int w = res.width;
      int h = res.height;

      shared_ptr<Commands::NewAnnotation> nA =
          std::make_shared<Commands::NewAnnotation>(project->getSession(),
                                                    this->object, this->frame,
                                                    x, y, w, h, 0.9f);
      commands.push_back(nA);
    }
  } catch (std::exception &e) {
  }

  return commands;
}

void MMOD::train() {
  if (!object) return;
  trainThread = std::thread([this] { this->trainWorker(); });
}

void MMOD::stop() {
  stopTraining = true;
  trainThread.join();
}

void MMOD::getImagesTrain() {
  assert(object);
  this->images_train.clear();
  this->boxes_train.clear();
  this->object->getAnnotations();

  for (auto annotation : this->object->getAnnotations()) {
    std::shared_ptr<AnnotatorLib::Annotation> a = annotation.second.lock();
    cv::Mat image =
        project->getImageSet()->getImage(a->getFrame()->getFrameNumber());

    dlib::matrix<dlib::rgb_pixel> dlibImage;
    dlib::assign_image(dlibImage, dlib::cv_image<dlib::rgb_pixel>(image));
    this->images_train.push_back(dlibImage);
    std::vector<dlib::mmod_rect> rects;
    long x1 = std::max(0L, (long)a->getX());
    long y1 = std::max(0L, (long)a->getY());
    long x2 =
        std::min((long)image.cols - 1L, (long)a->getX() + (long)a->getWidth());
    long y2 =
        std::min((long)image.rows - 1L, (long)a->getY() + long(a->getHeight()));
    dlib::rectangle rect(x1, y1, x2, y2);
    rects.push_back(dlib::mmod_rect(rect));
    this->boxes_train.push_back(rects);
  }
}

void MMOD::loadNet(std::string file) {
  try {
    dlib::deserialize(file) >> net;
  } catch (...) {
  }
}

void MMOD::saveNet(std::string file) {
  try {
    net.clean();
    dlib::serialize(file) << net;
  } catch (...) {
  }
}

void MMOD::trainWorker() {
  stopTraining = false;
  widget.setProgress(10);
  getImagesTrain();
  widget.setProgress(20);
  dlib::mmod_options options(boxes_train, 20 * 20, 10 * 10);//
  net = net_type(options);
  dlib::dnn_trainer<net_type> trainer(net);
  trainer.set_learning_rate(0.1);
  trainer.be_verbose();
  trainer.set_synchronization_file("mmod_sync", std::chrono::minutes(5));
  trainer.set_iterations_without_progress_threshold(300);
  widget.setProgress(30);
  std::vector<dlib::matrix<dlib::rgb_pixel>> mini_batch_samples;
  std::vector<std::vector<dlib::mmod_rect>> mini_batch_labels;

  dlib::random_cropper cropper;
  cropper.set_chip_dims(200, 200);
  cropper.set_min_object_size(0.2);
  dlib::rand rnd;
  // Run the trainer until the learning rate gets small.  This will probably
  // take several
  // hours.
  while (!stopTraining && trainer.get_learning_rate() >= 1e-4) {
    try {
      cropper(50, images_train, boxes_train, mini_batch_samples,
              mini_batch_labels);
      // We can also randomly jitter the colors and that often helps a detector
      // generalize better to new images.
      for (auto &&img : mini_batch_samples) disturb_colors(img, rnd);

      trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    } catch (dlib::impossible_labeling_error &e) {
      std::cout << this->getName().toStdString() << ": " << e.what()
                << std::endl;
    }
  }
  // wait for training threads to stop
  trainer.get_net();

  widget.setProgress(50);
  widget.setProgress(0);
}

cv::Rect MMOD::findObject() {
  dlib::matrix<dlib::rgb_pixel> dlibImage;
  dlib::assign_image(dlibImage,
                     dlib::cv_image<dlib::rgb_pixel>(this->frameImg));

  dlib::pyramid_up(dlibImage);

  std::vector<dlib::mmod_rect> dets = net(dlibImage);
  if (dets.size() < 1) return cv::Rect();

  dlib::rectangle found = dets[0];
  return cv::Rect(found.left(), found.top(), found.width(), found.height());
}

QPixmap MMOD::getImgCrop(shared_ptr<AnnotatorLib::Annotation> annotation,
                         int size) const {
  if (annotation == nullptr) return QPixmap();

  cv::Mat cropped = getImg(annotation);

  cropped.convertTo(cropped, CV_8U);
  cv::cvtColor(cropped, cropped, CV_BGR2RGB);

  QImage img((const unsigned char *)(cropped.data), cropped.cols, cropped.rows,
             cropped.step, QImage::Format_RGB888);

  QPixmap pim = QPixmap::fromImage(img);
  pim = pim.scaledToHeight(size);
  return pim;
}

cv::Mat MMOD::getImg(shared_ptr<Annotation> annotation) const {
  cv::Mat tmp = project->getImageSet()->getImage(
      annotation->getFrame()->getFrameNumber());

  float x = std::max(annotation->getX(), 0.f);
  float y = std::max(annotation->getY(), 0.f);
  float w = std::min(annotation->getWidth(), tmp.cols - x);
  float h = std::min(annotation->getHeight(), tmp.rows - y);

  cv::Rect rect(x, y, w, h);
  cv::Mat cropped;
  try {
    tmp(rect).copyTo(cropped);
  } catch (cv::Exception &e) {
    std::cout << e.what();
  }
  return cropped;
}
