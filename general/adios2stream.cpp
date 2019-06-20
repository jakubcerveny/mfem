/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * adios2stream.cpp : implementation of adios2stream functions
 *
 *  Created on: Feb 4, 2019
 *      Author: William F Godoy godoywf@ornl.gov
 */

#include "adios2stream.hpp"

#include "../fem/geom.hpp"
#include "../general/array.hpp"
#include "../mesh/element.hpp"
#include "../mesh/mesh.hpp"
#include "../fem/gridfunc.hpp"

#include <algorithm>

namespace mfem
{

namespace
{
// these functions might be included in adios2 upstream next release
template <class T>
adios2::Variable<T> SafeDefineVariable(adios2::IO io,
                                       const std::string& variable_name,
                                       const adios2::Dims& shape = adios2::Dims(),
                                       const adios2::Dims& start = adios2::Dims(),
                                       const adios2::Dims& count = adios2::Dims())
{
   adios2::Variable<T> variable = io.InquireVariable<T>(variable_name);
   if (variable)
   {
      if (variable.Count() != count)
      {
         variable.SetSelection({start, count});
      }
   }
   else
   {
      variable = io.DefineVariable<T>(variable_name, shape, start, count);
   }

   return variable;
}

template <class T>
adios2::Attribute<T> SafeDefineAttribute(adios2::IO io,
                                         const std::string& attribute_name,
                                         const T& value,
                                         const std::string& variable_name = "",
                                         const std::string separator = "/")
{
   adios2::Attribute<T> attribute = io.InquireAttribute<T>(attribute_name);
   if (attribute)
   {
      return attribute;
   }
   return io.DefineAttribute<T>(attribute_name, value, variable_name, separator );
}

template <class T>
adios2::Attribute<T> SafeDefineAttribute(adios2::IO io,
                                         const std::string& attribute_name,
                                         const T* values, const size_t size,
                                         const std::string& variable_name = "",
                                         const std::string separator = "/")
{
   adios2::Attribute<T> attribute = io.InquireAttribute<T>(attribute_name);
   if (attribute)
   {
      return attribute;
   }
   return io.DefineAttribute<T>(attribute_name, values, size, variable_name,
                                separator );
}

} //end empty namespace

// PUBLIC
#ifdef MFEM_USE_MPI
adios2stream::adios2stream(const std::string& name, const openmode mode,
                           MPI_Comm comm, const std::string engineType)
   : name(name),
     adios2_openmode(mode),
     adios(std::make_shared<adios2::ADIOS>(comm)),
     io(adios->DeclareIO(name))
{
   io.SetEngine(engineType);
}
#else
adios2stream::adios2stream(const std::string& name, const openmode mode,
                           const std::string engineType)
   : name(name),
     adios2_openmode(mode),
     adios(std::make_shared<adios2::ADIOS>()),
     io(adios->DeclareIO(name))
{
   io.SetEngine(engineType);
}
#endif

adios2stream::~adios2stream()
{
   if (engine)
   {
      SafeDefineAttribute<std::string>(io, "vtk.xml", VTKSchema() );
      engine.Close();
   }
}

void adios2stream::SetParameters(
   const std::map<std::string, std::string>& parameters)
{
   io.SetParameters(parameters);
}

void adios2stream::SetParameter(const std::string key,
                                const std::string value)
{
   io.SetParameter(key, value);
}

void adios2stream::BeginStep()
{
   if (!engine)
   {
      engine = io.Open(name, adios2::Mode::Write);
   }
   engine.BeginStep();
   active_step = true;
}

void adios2stream::EndStep()
{
   if (!engine || active_step == false)
   {
      const std::string message = "MFEM adios2stream error: calling EndStep "
                                  "on uninitialized step (need BeginStep)";
      mfem_error(message.c_str());
   }

   SafeDefineAttribute<std::string>(io, "vtk.xml", VTKSchema() );
   engine.EndStep();
   active_step = false;
}

size_t adios2stream::CurrentStep() const noexcept
{
   return engine.CurrentStep();
}

void adios2stream::Close()
{
   if (engine)
   {
      SafeDefineAttribute<std::string>(io, "vtk.xml", VTKSchema() );
      engine.Close();
   }
   if (adios)
   {
      adios.reset();
   }
}


// PROTECTED (accessible by friend class Mesh)
void adios2stream::Print(const Mesh& mesh, const mode print_mode)
{
   auto lf_DefineMeshMetadata = [this](const Mesh& mesh)
   {
      //check types are constant
      if (!IsConstantElementType(mesh.elements))
      {
         throw std::invalid_argument("MFEM::adios2stream ERROR: non-constant "
                                     " element types not yet implemented\n");
      }

      // format info
      SafeDefineAttribute<std::string>(io, "format", "MFEM ADIOS2 BP v0.1" );
      SafeDefineAttribute<std::string>(io, "format/version", "0.1" );
      std::string mesh_type = "Unknown";
      std::vector<std::string> viz_tools;
      viz_tools.reserve(2); //for now
      if (mesh.NURBSext)
      {
         mesh_type = "MFEM NURBS";
         viz_tools.push_back("NONE");
      }
      else if (mesh.ncmesh)
      {
         mesh_type = "MFEM mesh v1.1";
         viz_tools.push_back("NONE");
      }
      else
      {
         mesh_type = "MFEM mesh v1.0";
         viz_tools.push_back("Paraview: ADIOS2VTXReader");
         viz_tools.push_back("VTK: vtkADIOS2VTXReader.h");
      }

      SafeDefineAttribute<std::string>(io, "format/mfem_mesh", mesh_type );
      SafeDefineAttribute<std::string>(io, "format/viz_tools", viz_tools.data(),
                                       viz_tools.size() );



      // elements
      SafeDefineAttribute<uint32_t>(io, "dimension",
                                    static_cast<int32_t>(mesh.Dimension()) );
      SafeDefineVariable<uint32_t>(io,"NumOfElements", {adios2::LocalValueDim});
      const size_t nElements = static_cast<size_t>(mesh.GetNE());
      const size_t nElementVertices = static_cast<size_t>
                                      (mesh.elements[0]->GetNVertices());
      SafeDefineVariable<uint64_t>(io, "connectivity", {}, {}, {nElements, nElementVertices+1});
      SafeDefineVariable<uint32_t>(io, "types");

      // vertices
      SafeDefineVariable<uint32_t>(io,"NumOfVertices", {adios2::LocalValueDim});
      const GridFunction* grid_function = mesh.GetNodes();
      if (grid_function == nullptr)
      {
         const size_t nVertices = static_cast<size_t>(mesh.GetNV());
         const size_t spaceDim = static_cast<size_t>(mesh.SpaceDimension());
         //similar to Ordering::byVDIM
         SafeDefineVariable<double>( io, "vertices", {}, {}, {nVertices, spaceDim});
      }
      else
      {
         const size_t size = static_cast<size_t>(grid_function->Size());
         const FiniteElementSpace* fes = grid_function->FESpace();
         const size_t components = static_cast<size_t>(fes->GetVDim());
         const size_t tuples = size /components;

         if (fes->GetOrdering() == Ordering::byNODES)
         {
            SafeDefineVariable<double>(io,"vertices", {}, {}, {components, tuples} );
            SafeDefineAttribute<std::string>(io, "Ordering", "byNODES", "vertices" );
            ordering_by_node = true;
         }
         else
         {
            SafeDefineVariable<double>(io, "vertices", {}, {}, {tuples, components} );
            SafeDefineAttribute<std::string>(io, "Ordering", "byVDIM", "vertices" );
         }
      }
   };
   auto lf_PrintMeshData = [this](const Mesh& mesh)
   {
      // elements
      engine.Put("NumOfElements", static_cast<uint32_t>(mesh.GetNE()));

      const uint32_t vtkType =
         GLVISToVTKType(static_cast<int>(mesh.elements[0]->GetGeometryType()));
      engine.Put("types", vtkType);

      adios2::Variable<uint64_t> varConnectivity =
         io.InquireVariable<uint64_t>("connectivity");
      //zero-copy access to adios2 buffer to put non-contiguous to contiguous memory
      adios2::Variable<uint64_t>::Span spanConnectivity =
         engine.Put<uint64_t>(varConnectivity);

      size_t elementPosition = 0;
      for (int e = 0; e < mesh.GetNE(); ++e)
      {
         const int nVertices = mesh.elements[e]->GetNVertices();
         spanConnectivity[elementPosition] = nVertices;
         for (int v = 0; v < nVertices; ++v)
         {
            spanConnectivity[elementPosition + v + 1] =
               mesh.elements[e]->GetVertices()[v];
         }
         elementPosition += nVertices + 1;
      }

      // vertices
      engine.Put("NumOfVertices", static_cast<uint32_t>(mesh.GetNV()));

      if (mesh.GetNodes() == nullptr)
      {
         adios2::Variable<double> varVertices = io.InquireVariable<double>("vertices");
         //zero-copy access to adios2 buffer to put non-contiguous to contiguous memory
         adios2::Variable<double>::Span spanVertices = engine.Put(varVertices);

         for (int v = 0; v < mesh.GetNV(); ++v)
         {
            const int space_dim = mesh.SpaceDimension();
            for (int coord = 0; coord < space_dim; ++coord)
            {
               spanVertices[v * space_dim + coord] = mesh.vertices[v](coord);
            }
         }
      }
      else
      {
         const GridFunction* grid_function = mesh.GetNodes();
         grid_function->Print(*this, "vertices");
      }
   };

   // BODY OF FUNCTION STARTS HERE
   try
   {
      lf_DefineMeshMetadata(mesh);

      if (!engine) // if Engine is closed
      {
         engine = io.Open(name, adios2::Mode::Write);
      }

      lf_PrintMeshData(mesh);

      if (print_mode == mode::sync)
      {
         engine.PerformPuts();
      }
   }
   catch (std::exception& e)
   {
      const std::string warning =
         "MFEM: adios2stream exception caught, invalid bp dataset: " + name +
         "," + e.what();
      mfem_warning( warning.c_str());
   }
}

void adios2stream::Save(const GridFunction& grid_function,
                        const std::string& variable_name, const data_type type)
{
   auto lf_BoolParameter = [&](const std::string key,
                               const bool default_value) -> bool
   {
      const adios2::Params& parameters = io.Parameters();
      auto it = parameters.find(key);
      if (it != parameters.end())
      {
         std::string value = it->second;
         std::transform(value.begin(), value.end(), value.begin(), ::tolower);
         if (value == "on" || value == "true")
         {
            return true;
         }
         else if ( value == "off" || value == "false")
         {
            return false;
         }
      }
      return default_value;
   };

   const FiniteElementSpace* fes = grid_function.FESpace();

   auto lf_SafeDefine = [&](const std::string& variable_name,
                            const size_t tuples, const size_t components)
   {
      adios2::Variable<double> var = io.InquireVariable<double>(variable_name);
      if (!var)
      {
         if (components == 1 && type == adios2stream::data_type::point_data)
         {
            io.DefineVariable<double>(variable_name, {}, {}, {tuples*components});
         }
         else
         {
            const adios2::Dims count = (fes->GetOrdering() == Ordering::byNODES) ?
                                       adios2::Dims{components, tuples} :
                                       adios2::Dims{tuples, components};
            io.DefineVariable<double>(variable_name, {}, {}, count);
         }
         SafeDefineAttribute<std::string>(io, "FiniteElementSpace",
                                          std::string(fes->FEColl()->Name()),
                                          variable_name);
      }
   };

   //BODY OF FUNCTION STARTS HERE
   const bool full_data = lf_BoolParameter("FullData", true);
   const bool refined_data = lf_BoolParameter("RefinedData", false);

   if (!full_data && !refined_data)
   {
      return;
   }

   if (refined_data)
   {
      Mesh *mesh = fes->GetMesh();
      const size_t components = static_cast<size_t>(fes->GetVDim());
      const size_t tuples = static_cast<size_t>(mesh->GetNV());
      lf_SafeDefine(variable_name, tuples, components);
      if (type == adios2stream::data_type::point_data)
      {
         point_data_variables.insert(variable_name);
      }

      RefinedGeometry* refined_geometry;
      DenseMatrix transform;

      //zero-copy access to adios2 buffer to put non-contiguous to contiguous memory
      adios2::Variable<double> variable = io.InquireVariable<double>(variable_name);
      adios2::Variable<double>::Span span = engine.Put<double>(variable);

      size_t offset = 0;
      if (components == 1)
      {
         Vector scalar;
         for (int e = 0; e < mesh->GetNE(); e++)
         {
            refined_geometry = GlobGeometryRefiner.Refine(
                                  mesh->GetElementBaseGeometry(e), 1, 1);

            grid_function.GetValues(e, refined_geometry->RefPts, scalar, transform);
            //for now since all elements are constant
            for (int i = 0; i < scalar.Size(); ++i)
            {
               span[offset + i] = scalar(i);
            }
            offset += static_cast<size_t>(scalar.Size());
         }
      }
      else
      {
         DenseMatrix vector;
         for (int e = 0; e < mesh->GetNE(); ++e)
         {
            refined_geometry = GlobGeometryRefiner.Refine(
                                  mesh->GetElementBaseGeometry(e), 1, 1);
            grid_function.GetVectorValues(e, refined_geometry->RefPts, vector, transform);

            for (int i = 0; i < vector.Width(); ++i)
            {
               for (int j = 0; j < vector.Height(); ++j)
               {
                  span[offset +  i*vector.Height() + j] = vector(j, i);
               }
            }
            offset += static_cast<size_t>(vector.Weight()*vector.Height());
         }
      }
   }

   if (full_data)
   {
      const size_t size = static_cast<size_t>(grid_function.Size());
      const size_t components = static_cast<size_t>(fes->GetVDim());
      const size_t tuples = size /components;
      lf_SafeDefine(variable_name +"/full", tuples, components);
      if (type == adios2stream::data_type::point_data)
      {
         point_data_variables.insert(variable_name+"/full");
      }
      // calls Vector::Print
      grid_function.Print(*this, variable_name+"/full");
   }
}

// PRIVATE
int32_t adios2stream::GLVISToVTKType(
   const int glvisType) const noexcept
{
   uint32_t vtkType = 0;
   switch (glvisType)
   {
      case Geometry::Type::POINT:
         vtkType = 1;
         break;
      case Geometry::Type::SEGMENT:
         vtkType = 3;
         break;
      case Geometry::Type::TRIANGLE:
         vtkType = 5;
         break;
      case Geometry::Type::SQUARE:
         vtkType = 8;
         //vtkType = 9;
         break;
      case Geometry::Type::TETRAHEDRON:
         vtkType = 10;
         break;
      case Geometry::Type::CUBE:
         vtkType = 11;
         break;
      case Geometry::Type::PRISM:
         vtkType = 13;
         break;
      default:
         vtkType = 0;
         break;
   }
   return vtkType;
}

bool adios2stream::IsConstantElementType(const Array<Element*>& elements ) const
noexcept
{
   bool isConstType = true;
   const Geometry::Type type = elements[0]->GetGeometryType();

   for (int e = 1; e < elements.Size(); ++e)
   {
      if (type != elements[e]->GetGeometryType())
      {
         isConstType = false;
         break;
      }
   }
   return isConstType;
}

std::string adios2stream::VTKSchema() const noexcept
{
   // might be extended in the future, from vtk docs:
   // AOS = Array of Structs (VTK default) = Order::byVDim  XYZ XYZ XYZ
   // SOA = Struct of Arrays = Order::ByNode  XXXX YYYY ZZZZ
   const std::string ordering = ordering_by_node? "SOA" : "AOS";

   std::string vtkSchema = R"(
<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="NumOfVertices" NumberOfCells="NumOfElements">
      <Points>
        <DataArray Name="vertices" Ordering=")";

   vtkSchema += ordering + "\"/>\n";

   vtkSchema += R"(
      </Points>
      <Cells>
        <DataArray Name="connectivity" />
        <DataArray Name="types" />
      </Cells>
      <PointData>)";

   if (point_data_variables.empty())
   {
      vtkSchema += "\n";
   }
   else
   {
      for (const std::string& point_datum : point_data_variables )
      {
         vtkSchema += "        <DataArray Name=\"" + point_datum +"\"/>\n";
      }
   }
   vtkSchema += R"(
       </PointData>
       </Piece>
     </UnstructuredGrid>
   </VTKFile>)";

   return vtkSchema;
}

adios2::Mode adios2stream::ToADIOS2Mode(const adios2stream::openmode mode) const
noexcept
{
   adios2::Mode adios2Mode = adios2::Mode::Undefined;
   switch (mode)
   {
      case adios2stream::openmode::out:
         adios2Mode = adios2::Mode::Write;
         break;
      case adios2stream::openmode::in:
         adios2Mode = adios2::Mode::Read;
         break;
      default:
         const std::string message = "MFEM adios2stream ERROR: only "
                                     "openmode::out and openmode::in "
                                     " are valid, in call to adios2stream constructor";
         mfem_error(message.c_str());
   }
   return adios2Mode;
}

}  // end namespace mfem
