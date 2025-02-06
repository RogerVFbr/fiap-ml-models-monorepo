import time
import ssl
import inspect

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# def time_it(func):
#     def wrapper(*args, **kwargs):
#         print()
#         print(f"=====> Executing '{func.__name__}' ...")
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         execution_time = end_time - start_time
#
#         minutes = int(execution_time // 60)
#         seconds = int(execution_time % 60)
#         milliseconds = int((execution_time - int(execution_time)) * 1000)
#
#         if minutes > 0:
#             time_str = f'{minutes}m {seconds}s {milliseconds}s'
#         else:
#             time_str = f'{seconds}s {milliseconds}ms'
#
#         print()
#         print(f"<===== '{func.__name__}' elapsed time: {time_str}.")
#
#         return result
#     return wrapper

def time_it(func):
    def wrapper(*args, **kwargs):
        # Detect the class name
        class_name = None
        file_name = None
        if len(args) > 0:
            instance = args[0]
            if inspect.isclass(instance.__class__):
                class_name = instance.__class__.__name__
                file_name = inspect.getfile(instance.__class__).split('/')[-1].split('.')[0]

        print()

        # Print the start message
        if class_name and file_name:
            print(f"=====> [{file_name}.{class_name}.{func.__name__}] Executing ...")
        else:
            print(f"=====> [N/A.{func.__name__}] Executing ...")

        # Measure the execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Format the execution time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        milliseconds = int((execution_time - int(execution_time)) * 1000)

        if minutes > 0:
            time_str = f'{minutes}m {seconds}s {milliseconds}ms'
        else:
            time_str = f'{seconds}s {milliseconds}ms'

        # Print the end message
        if class_name and file_name:
            print(f"<===== [{file_name}.{class_name}.{func.__name__}] Elapsed: {time_str}.")
        else:
            print(f"<===== [N/A.{func.__name__}] Elapsed: {time_str}.")


        return result
    return wrapper