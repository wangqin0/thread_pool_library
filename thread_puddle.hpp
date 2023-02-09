#pragma once

#include <atomic>             // std::atomic
#include <condition_variable> // std::condition_variable
#include <exception>          // std::current_exception
#include <functional>         // std::bind, std::function, std::invoke
#include <future>             // std::future, std::promise
#include <memory>             // std::make_shared, std::make_unique, std::shared_ptr, std::unique_ptr
#include <mutex>              // std::mutex, std::scoped_lock, std::unique_lock
#include <queue>              // std::queue
#include <thread>             // std::thread
#include <type_traits>        // std::common_type_t, std::decay_t, std::invoke_result_t, std::is_void_v
#include <utility>            // std::forward, std::move, std::swap

namespace TP
{

using concurrency_t = std::invoke_result_t<decltype(std::thread::hardware_concurrency)>;


class ThreadPuddle
{
public:

    explicit ThreadPuddle(const concurrency_t thread_count_ = 0) : threadCount(determineThreadCount(thread_count_)), threads(std::make_unique<std::thread[]>(
            determineThreadCount(thread_count_)))
    {
        createThreads();
    }


    ~ThreadPuddle()
    {
        waitForTasks();
        destroyThreads();
    }


    concurrency_t getThreadCount() const
    {
        return threadCount;
    }


    template <typename F, typename T1, typename T2, typename T = std::common_type_t<T1, T2>>
    void pushLoop(T1 first_index_, T2 index_after_last_, F&& loop, size_t num_blocks = 0)
    {
        T first_index = static_cast<T>(first_index_);
        T index_after_last = static_cast<T>(index_after_last_);
        if (num_blocks == 0)
            num_blocks = threadCount;
        if (index_after_last < first_index)
            std::swap(index_after_last, first_index);
        auto total_size = static_cast<size_t>(index_after_last - first_index);
        auto block_size = static_cast<size_t>(total_size / num_blocks);
        if (block_size == 0)
        {
            block_size = 1;
            num_blocks = (total_size > 1) ? total_size : 1;
        }
        if (total_size > 0)
        {
            for (size_t i = 0; i < num_blocks; ++i)
                pushTask(std::forward<F>(loop), static_cast<T>(i * block_size) + first_index,
                         (i == num_blocks - 1) ? index_after_last : (static_cast<T>((i + 1) * block_size) +
                                                                     first_index));
        }
    }


    template <typename F, typename T>
    void pushLoop(const T index_after_last, F&& loop, const size_t num_blocks = 0)
    {
        pushLoop(0, index_after_last, std::forward<F>(loop), num_blocks);
    }


    template <typename F, typename... A>
    void pushTask(F&& task, A&&... args)
    {
        std::function<void()> task_function = std::bind(std::forward<F>(task), std::forward<A>(args)...);
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            tasks.push(task_function);
        }
        ++tasks_total;
        task_available_cv.notify_one();
    }


    template <typename F, typename... A, typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>
    std::future<R> submit(F&& task, A&&... args)
    {
        std::function<R()> task_function = std::bind(std::forward<F>(task), std::forward<A>(args)...);
        std::shared_ptr<std::promise<R>> task_promise = std::make_shared<std::promise<R>>();
        pushTask(
                [task_function, task_promise] {
                    try {
                        if constexpr (std::is_void_v<R>) {
                            std::invoke(task_function);
                            task_promise->set_value();
                        } else {
                            task_promise->set_value(std::invoke(task_function));
                        }
                    }
                    catch (...) {
                        try {
                            task_promise->set_exception(std::current_exception());
                        }
                        catch (...) {
                        }
                    }
                });
        return task_promise->get_future();
    }


    void waitForTasks()
    {
        waiting = true;
        std::unique_lock<std::mutex> tasks_lock(tasks_mutex);
        task_done_cv.wait(tasks_lock, [this] { return (tasks_total == 0); });
        waiting = false;
    }

private:

    void createThreads()
    {
        running = true;
        for (concurrency_t i = 0; i < threadCount; ++i)
        {
            threads[i] = std::thread(&ThreadPuddle::worker, this);
        }
    }


    void destroyThreads()
    {
        running = false;
        task_available_cv.notify_all();
        for (concurrency_t i = 0; i < threadCount; ++i)
        {
            threads[i].join();
        }
    }


    static concurrency_t determineThreadCount(const concurrency_t thread_count_)
    {
        if (thread_count_ > 0)
            return thread_count_;
        else
        {
            if (std::thread::hardware_concurrency() > 0)
                return std::thread::hardware_concurrency();
            else
                return 1;
        }
    }


    void worker()
    {
        while (running)
        {
            std::function<void()> task;
            std::unique_lock<std::mutex> tasks_lock(tasks_mutex);
            task_available_cv.wait(tasks_lock, [this] { return !tasks.empty() || !running; });
            if (running)
            {
                task = std::move(tasks.front());
                tasks.pop();
                tasks_lock.unlock();
                task();
                tasks_lock.lock();
                --tasks_total;
                if (waiting)
                    task_done_cv.notify_one();
            }
        }
    }


    std::atomic<bool> running = false;


    std::condition_variable task_available_cv = {};


    std::condition_variable task_done_cv = {};


    std::queue<std::function<void()>> tasks = {};


    std::atomic<size_t> tasks_total = 0;


    mutable std::mutex tasks_mutex = {};


    concurrency_t threadCount = 0;


    std::unique_ptr<std::thread[]> threads = nullptr;


    std::atomic<bool> waiting = false;
};

} // namespace TP
