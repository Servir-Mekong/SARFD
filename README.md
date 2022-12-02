# SARFDAS
This readme describes the software routines for SARDAS Palawan

Developer: Ate Poortinga (apoortinga@sig-gis.com)
Date: 12/2022


DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

alerts.py : convolutitional neural network to train network on forest alerts using sentinel-1 timeseries information

running_sampling.py : export tensorflow records from GEE 

cronModel.py : script to run inference for Palawan Island

setDate.py : add date stamp to images

vectorizeData.py : convert alerts to feature collection 

mergeVectorAlerts.py : merge data into a single feature collection 
