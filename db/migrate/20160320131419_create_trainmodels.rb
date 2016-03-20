class CreateTrainmodels < ActiveRecord::Migration
  def change
    create_table :trainmodels do |t|
      t.integer :modid
      t.text :moddesc
      t.string :modname
      t.string :trainfile
      t.string :testfile
      t.text :addtnl
      t.boolean :modcheck

      t.timestamps null: false
    end
  end
end
