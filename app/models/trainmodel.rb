class Trainmodel < ActiveRecord::Base
	mount_uploader :trainfile, CsvtrainUploader
	mount_uploader :testfile, CsvtestUploader
end
