class TrainmodelsController < ApplicationController
  before_action :set_trainmodel, only: [:show, :edit, :update, :destroy]

  # GET /trainmodels
  # GET /trainmodels.json
  def index
    @trainmodels = Trainmodel.all
  end

  # GET /trainmodels/1
  # GET /trainmodels/1.json
  def show
	rowarray = Array.new
    	myfile = Trainmodel.find(params[:id]).trainfile
	c = CSV.open(myfile.path)
    	c.first(10).each do |row|
	      	rowarray << row
      		@rowarraydisp = rowarray
    	end
  end

  # GET /trainmodels/new
  def new
    @trainmodel = Trainmodel.new
  end

  # GET /trainmodels/1/edit
  def edit
  end

  def trainbutton
	%x(python churncode.py > donkey)
	puts "python"
	respond_to do |format|
	
		format.html { redirect_to trainmodels_path , notice:'The model was trained successfully.'}

	end
  end
	
  def testbutton
	%x(echo 1)
	puts "test button fucntion working"
	respond_to do |format|
		format.html { redirect_to trainmodels_path , notice:'the test data was uploaded and is testing' }
	end
  end 
 # POST /trainmodels
  # POST /trainmodels.json
  def create
    @trainmodel = Trainmodel.new(trainmodel_params)

    respond_to do |format|
      if @trainmodel.save
        format.html { redirect_to @trainmodel, notice: 'Trainmodel was successfully created.' }
        format.json { render :show, status: :created, location: @trainmodel }
      else
        format.html { render :new }
        format.json { render json: @trainmodel.errors, status: :unprocessable_entity }
      end
    end
  end

  # PATCH/PUT /trainmodels/1
  # PATCH/PUT /trainmodels/1.json
  def update
    respond_to do |format|
      if @trainmodel.update(trainmodel_params)
        format.html { redirect_to @trainmodel, notice: 'Trainmodel was successfully updated.' }
        format.json { render :show, status: :ok, location: @trainmodel }
      else
        format.html { render :edit }
        format.json { render json: @trainmodel.errors, status: :unprocessable_entity }
      end
    end
  end

  # DELETE /trainmodels/1
  # DELETE /trainmodels/1.json
  def destroy
    @trainmodel.destroy
    respond_to do |format|
      format.html { redirect_to trainmodels_url, notice: 'Trainmodel was successfully destroyed.' }
      format.json { head :no_content }
    end
  end

  private
    # Use callbacks to share common setup or constraints between actions.
    def set_trainmodel
      @trainmodel = Trainmodel.find(params[:id])
    end

    # Never trust parameters from the scary internet, only allow the white list through.
    def trainmodel_params
      params.require(:trainmodel).permit(:modid, :moddesc, :modname, :trainfile, :testfile, :addtnl, :modcheck)
    end
end
