Rails.application.routes.draw do
  resources :trainmodels
  root to: 'visitors#index'
  devise_for :users
  resources :users
  get '/publish' => 'trainmodels#trainbutton'
  get '/testdata' => 'trainmodels#testbutton'
end
