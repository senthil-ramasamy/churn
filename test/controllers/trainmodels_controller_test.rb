require 'test_helper'

class TrainmodelsControllerTest < ActionController::TestCase
  setup do
    @trainmodel = trainmodels(:one)
  end

  test "should get index" do
    get :index
    assert_response :success
    assert_not_nil assigns(:trainmodels)
  end

  test "should get new" do
    get :new
    assert_response :success
  end

  test "should create trainmodel" do
    assert_difference('Trainmodel.count') do
      post :create, trainmodel: { addtnl: @trainmodel.addtnl, modcheck: @trainmodel.modcheck, moddesc: @trainmodel.moddesc, modid: @trainmodel.modid, modname: @trainmodel.modname, testfile: @trainmodel.testfile, trainfile: @trainmodel.trainfile }
    end

    assert_redirected_to trainmodel_path(assigns(:trainmodel))
  end

  test "should show trainmodel" do
    get :show, id: @trainmodel
    assert_response :success
  end

  test "should get edit" do
    get :edit, id: @trainmodel
    assert_response :success
  end

  test "should update trainmodel" do
    patch :update, id: @trainmodel, trainmodel: { addtnl: @trainmodel.addtnl, modcheck: @trainmodel.modcheck, moddesc: @trainmodel.moddesc, modid: @trainmodel.modid, modname: @trainmodel.modname, testfile: @trainmodel.testfile, trainfile: @trainmodel.trainfile }
    assert_redirected_to trainmodel_path(assigns(:trainmodel))
  end

  test "should destroy trainmodel" do
    assert_difference('Trainmodel.count', -1) do
      delete :destroy, id: @trainmodel
    end

    assert_redirected_to trainmodels_path
  end
end
