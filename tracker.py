
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from activator import activate

import matplotlib
from matplotlib import pyplot as plt
import numpy
import time
from os import path
import tempfile
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import pandas as pd
import pandas
import pystore
from data_extractor import Hardware

pystore.set_path('./data')
store = pystore.store('testdatastore')
collection = store.collection('sample.EOD')
df = collection.item('CPU-Util').to_pandas()

_PATH = path.dirname(__file__)
_CSV_FILE = path.join(_PATH, 'test-dataset.csv')



def bound_forecasts_between_0_and_100(ndarray):

  return numpy.clip(ndarray, 0, 100)

def upload_data(data_name,num_sample=100) :

  util = Hardware()
  util_df = util.helper(12,save=0)
  util_df.insert(0,'time',util_df.index,True)
  #print(util_df.head())
  collection.write(data_name,util_df,metadata={'Source': data_name},overwrite=True)


def get_data(data_name) :
  return collection.item(data_name).to_pandas()

def multiple_timeseries_forecast(
    data_name='CPU-Util', export_directory=None, training_steps=500):

  estimator = tf.contrib.timeseries.StructuralEnsembleRegressor(
      periodicities=[], num_features=4)

  df = get_data(data_name)
  #print("\n\n\n--------------------------",df.iloc[:,0].values,"\n\n",numpy.asarray(df.iloc[:,2:].values,dtype='float32'),"\n------------------------------\n\n\n")

  data = {tf.contrib.timeseries.TrainEvalFeatures.TIMES: df.iloc[:,0].values,tf.contrib.timeseries.TrainEvalFeatures.VALUES: numpy.asarray(numpy.asarray(df.iloc[:,1:].values,dtype='float32'),dtype='int32')}
  np_reader = tf.contrib.timeseries.NumpyReader(data)


  # reader = tf.contrib.timeseries.CSVReader(
  #     csv_file_name,
  #     skip_header_lines=1,
  #     column_names=((tf.contrib.timeseries.TrainEvalFeatures.TIMES,)
  #                   + (tf.contrib.timeseries.TrainEvalFeatures.VALUES,) * 4))
  train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
  np_reader, batch_size=4, window_size=10)

  print("\n\ntrain_input_fn: ",train_input_fn,"\n\n")
  estimator.train(input_fn=train_input_fn, steps=training_steps)
  evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(np_reader)
  current_state = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
  values = [current_state["observed"]]
  times = [current_state[tf.contrib.timeseries.FilteringResults.TIMES]]
  if export_directory is None:
    export_directory = tempfile.mkdtemp()
  input_receiver_fn = estimator.build_raw_serving_input_receiver_fn()
  export_location = estimator.export_savedmodel(
      export_directory, input_receiver_fn)
  with tf.Graph().as_default():
    numpy.random.seed(1)
    with tf.Session() as session:
      signatures = tf.saved_model.loader.load(
          session, [tf.saved_model.tag_constants.SERVING], export_location)
      for _ in range(100):
        current_prediction = (
            tf.contrib.timeseries.saved_model_utils.predict_continuation(
                continue_from=current_state, signatures=signatures,
                session=session, steps=1))
        next_sample = numpy.random.multivariate_normal(
            mean=numpy.squeeze(current_prediction["mean"], axis=(0, 1)),
            cov=numpy.squeeze(current_prediction["covariance"], axis=(0, 1)))
        filtering_features = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: current_prediction[
                tf.contrib.timeseries.FilteringResults.TIMES],
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: next_sample[
                None, None, :]}
        current_state = (
            tf.contrib.timeseries.saved_model_utils.filter_continuation(
                continue_from=current_state,
                session=session,
                signatures=signatures,
                features=filtering_features))
        values.append(next_sample[None, None, :])
        times.append(current_state["times"])
  past_and_future_values = numpy.squeeze(numpy.concatenate(values, axis=1), axis=0)
  past_and_future_timesteps = numpy.squeeze(numpy.concatenate(times, axis=1), axis=0)
  return past_and_future_timesteps, past_and_future_values


def main(unused_argv):

  #upload_data('CPU-Util1')

  startTime = time.time()
  past_and_future_timesteps, past_and_future_values = multiple_timeseries_forecast(data_name='CPU-Util')
  endTime = time.time()

  print('len(past_and_future_timesteps)', len(past_and_future_timesteps))
  print('len(past_and_future_values)', len(past_and_future_values))
  #print('type(past_and_future_timesteps)', type(past_and_future_timesteps))    #<class 'numpy.ndarray'>
  #print('type(past_and_future_values)', type(past_and_future_values))      #<class 'numpy.ndarray'>
  #print('past_and_future_timesteps.shape', past_and_future_timesteps.shape)
  #print('past_and_future_values.shape', past_and_future_values.shape)

  #print('past_and_future_timesteps[995:1005]:', past_and_future_timesteps[995:1005])

  print('\n##### first 1000 samples #####')
  print('min value:', numpy.amin(past_and_future_values[:999]), 'at ', numpy.unravel_index(past_and_future_values[:999].argmin(), past_and_future_values[:999].shape))    #returns
  print('max value:', numpy.amax(past_and_future_values[:999]), 'at ', numpy.unravel_index(past_and_future_values[:999].argmax(), past_and_future_values[:999].shape))    #returns

  print('\n##### 100 future forecast samples #####')

  print('min value:', numpy.amin(past_and_future_values[999:]), 'at ', numpy.unravel_index(past_and_future_values[999:].argmin(), past_and_future_values[999:].shape))    #returns

  print('max value:', numpy.amax(past_and_future_values[999:]), 'at ', numpy.unravel_index(past_and_future_values[999:].argmax(), past_and_future_values[999:].shape))    #returns

  print('past_and_future_values[999:].shape:', past_and_future_values[999:].shape)

  print('\nAll future 100 forecast values:\n', past_and_future_values[999:])

  print('Now bounding forecasts between 0 and 100 since this is a system resource utilization problem.')

  #bound forecasts between 0 and 100
  past_and_future_values[999:] = bound_forecasts_between_0_and_100(past_and_future_values[999:])
  print('Done! Now displaying a visualization of 1000 past timesteps, and 100 future timesteps with forecast values for multiple features.')

  # Show where sampling starts on the plot
  plt.axvline(1000, linestyle="dotted")
  plt.plot(past_and_future_timesteps, past_and_future_values)
  plt.title('Simultaneous forecast of multiple time series features')
  plt.xlabel("Timesteps")
  plt.ylabel("Units")
  #handles, labels = ax.get_legend_handles_labels()
  #ax.legend(handles, labels)
  plt.show()
  activate(past_and_future_values[900:,0])

  #print elapsed time in hh:mm:ss format
  hours, rem = divmod(endTime-startTime, 3600)
  minutes, seconds = divmod(rem, 60)
  print("Time elapsed: {:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))
  print('finished running tftf-sep28-2018-multivariate-tensorflow-timeseries-forecasting.py')
  numpy.savetxt('timeseries-output.csv', past_and_future_values, delimiter=",")
  print('done writing output to timeseries-output.csv')

if __name__ == "__main__":
  print(tf.__version__)
  #print(tf.contrib.timeseries.TrainEvalFeatures.TIMES,tf.contrib.timeseries.TrainEvalFeatures.VALUES)
  tf.compat.v1.app.run(main=None,argv=None)
  #tf.app.run(main=main)
