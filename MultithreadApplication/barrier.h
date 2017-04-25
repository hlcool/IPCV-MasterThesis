#ifndef BARRIER_H
#define BARRIER_H

#include <QMutex>
#include <QWaitCondition>
#include <QSharedPointer>

// Data "pimpl" class (not to be used directly)
class BarrierData
{
public:
    BarrierData(int count) : count(count) {}

    void wait() {
        mutex.lock();
        --count;
        if (count > 0){
            condition.wait(&mutex);
        }
        else{
            condition.wakeAll();
            count = 3;
        }
        mutex.unlock();
    }
private:
    Q_DISABLE_COPY(BarrierData)
    int count;
    QMutex mutex;
    QWaitCondition condition;
};

class Barrier {
public:
    // Create a barrier that will wait for count threads
    Barrier(int count) : d(new BarrierData(count)) {}
    void wait() {
        d->wait();
    }

private:
    QSharedPointer<BarrierData> d;
};

#endif // BARRIER_H
