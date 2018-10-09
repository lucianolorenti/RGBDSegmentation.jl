using LibPQ

using SearchLight, SearchLight.Database, Nullables
import SearchLight.Migrations: create_table, column, primary_key, add_index, drop_table
SearchLight.Database.setup_adapter("PostgreSQLDatabaseAdapter")
Database.connect( Dict(
  "host"     => "localhost",
  "password" => "plaza113",
  "username" => "luciano",
  "port"     => 5432,
  "database" => "specia_results",
"adapter"  => "PostgreSQL",
))
mutable struct Test <: AbstractModel
    _table_name::String
    _id::String
    id::SearchLight.DbId
    dataset::String
    image_number::Integer
    results::Dict
end
function Test(;
         id = Nullable{SearchLight.DbId}(),
         dataset::String = "",
         image_number::Integer = 0,
         results::Dict = Dict()
              )
    return Test("tests", "id", id, dataset, image_number, results)
end

function up()
  create_table(:tests) do
    [
      primary_key()
      column(:dataset, :string)
      column(:image_number, :integer)
      column(:results, :dict)
    ]
  end
end

function down()
  drop_table(:tests)
end


#SearchLight.db_init()
#up()
t=Test(dataset = "NYU", image_number=1, results=Dict("FP"=>0))
save!(t)
