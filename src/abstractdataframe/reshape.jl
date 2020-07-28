"""
    stack(df::AbstractDataFrame, [measure_vars], [id_vars];
          variable_name=:variable, value_name=:value,
          view::Bool=false, variable_eltype::Type=String)

Stack a data frame `df`, i.e. convert it from wide to long format.

Return the long-format `DataFrame` with: columns for each of the `id_vars`,
column `variable_name` (`:value` by default)
holding the values of the stacked columns (`measure_vars`), and
column `variable_name` (`:variable` by default) a vector holding
the name of the corresponding `measure_vars` variable.

If `view=true` then return a stacked view of a data frame (long format).
The result is a view because the columns are special `AbstractVectors`
that return views into the original data frame.


# Arguments
- `df` : the AbstractDataFrame to be stacked
- `measure_vars` : the columns to be stacked (the measurement variables),
  as a column selector ($COLUMNINDEX_STR; $MULTICOLUMNINDEX_STR).
  If neither `measure_vars` or `id_vars` are given, `measure_vars`
  defaults to all floating point columns.
- `id_vars` : the identifier columns that are repeated during stacking,
  as a column selector ($COLUMNINDEX_STR; $MULTICOLUMNINDEX_STR).
  Defaults to all variables that are not `measure_vars`
- `variable_name` : the name (`Symbol` or string) of the new stacked column that
  shall hold the names of each of `measure_vars`
- `value_name` : the name (`Symbol` or string) of the new stacked column containing
  the values from each of `measure_vars`
- `view` : whether the stacked data frame should be a view rather than contain
  freshly allocated vectors.
- `variable_eltype` : determines the element type of column `variable_name`.
  If `variable_eltype=Symbol` it is a vector of `Symbol`,
  and if `variable_eltype=String` a vector of `String` is produced.


# Examples
```julia
d1 = DataFrame(a = repeat([1:3;], inner = [4]),
               b = repeat([1:4;], inner = [3]),
               c = randn(12),
               d = randn(12),
               e = map(string, 'a':'l'))

d1s = stack(d1, [:c, :d])
d1s2 = stack(d1, [:c, :d], [:a])
d1m = stack(d1, Not([:a, :b, :e]))
d1s_name = stack(d1, Not([:a, :b, :e]), variable_name=:somemeasure)
```
"""
function stack(df::AbstractDataFrame,
               measure_vars = findall(col -> eltype(col) <: Union{AbstractFloat, Missing},
                                      eachcol(df)),
               id_vars = Not(measure_vars);
               variable_name::SymbolOrString=:variable,
               value_name::SymbolOrString=:value, view::Bool=false,
               variable_eltype::Type=String)
    variable_name_s = Symbol(variable_name)
    value_name_s = Symbol(value_name)
    # getindex from index returns either Int or AbstractVector{Int}
    mv_tmp = index(df)[measure_vars]
    ints_measure_vars = mv_tmp isa Int ? [mv_tmp] : mv_tmp
    idv_tmp = index(df)[id_vars]
    ints_id_vars = idv_tmp isa Int ? [idv_tmp] : idv_tmp
    if view
        return _stackview(df, ints_measure_vars, ints_id_vars,
                          variable_name=variable_name_s,
                          value_name=value_name_s,
                          variable_eltype=variable_eltype)
    end
    N = length(ints_measure_vars)
    cnames = _names(df)[ints_id_vars]
    push!(cnames, variable_name_s)
    push!(cnames, value_name_s)
    if variable_eltype === Symbol
        catnms = _names(df)[ints_measure_vars]
    elseif variable_eltype === String
        catnms = PooledArray(names(df, ints_measure_vars))
    else
        throw(ArgumentError("`variable_eltype` keyword argument accepts only " *
                            "`String` or `Symbol` as a value."))
    end
    return DataFrame(AbstractVector[[repeat(df[!, c], outer=N) for c in ints_id_vars]..., # id_var columns
                                    repeat(catnms, inner=nrow(df)),                       # variable
                                    vcat([df[!, c] for c in ints_measure_vars]...)],      # value
                     cnames, copycols=false)
end

function _stackview(df::AbstractDataFrame, measure_vars::AbstractVector{Int},
                    id_vars::AbstractVector{Int}; variable_name::Symbol,
                    value_name::Symbol, variable_eltype::Type)
    N = length(measure_vars)
    cnames = _names(df)[id_vars]
    push!(cnames, variable_name)
    push!(cnames, value_name)
    if variable_eltype <: Symbol
        catnms = _names(df)[measure_vars]
    elseif variable_eltype <: String
        catnms = names(df, measure_vars)
    else
        throw(ArgumentError("`variable_eltype` keyword argument accepts only " *
                            "`String` or `Symbol` as a value."))
    end
    return DataFrame(AbstractVector[[RepeatedVector(df[!, c], 1, N) for c in id_vars]..., # id_var columns
                                    RepeatedVector(catnms, nrow(df), 1),                  # variable
                                    StackedVector(Any[df[!, c] for c in measure_vars])],  # value
                     cnames, copycols=false)
end

"""
    unstack(df::AbstractDataFrame, rowkeys, colkey, value; renamecols::Function=identity)
    unstack(df::AbstractDataFrame, colkey, value; renamecols::Function=identity)
    unstack(df::AbstractDataFrame; renamecols::Function=identity)

Unstack data frame `df`, i.e. convert it from long to wide format.

If `colkey` contains `missing` values then they will be skipped and a warning
will be printed.

If combination of `rowkeys` and `colkey` contains duplicate entries then last
`value` will be retained and a warning will be printed.

# Arguments
- `df` : the AbstractDataFrame to be unstacked
- `rowkeys` : the columns with a unique key for each row, if not given,
  find a key by grouping on anything not a `colkey` or `value`.
  Can be any column selector ($COLUMNINDEX_STR; $MULTICOLUMNINDEX_STR).
- `colkey` : the column ($COLUMNINDEX_STR) holding the column names in wide format,
  defaults to `:variable`
- `value` : the value column ($COLUMNINDEX_STR), defaults to `:value`
- `renamecols` : a function called on each unique value in `colkey` which must
                 return the name of the column to be created (typically as a string
                 or a `Symbol`). Duplicate names are not allowed.


# Examples
```julia
wide = DataFrame(id = 1:12,
                 a  = repeat([1:3;], inner = [4]),
                 b  = repeat([1:4;], inner = [3]),
                 c  = randn(12),
                 d  = randn(12))

long = stack(wide)
wide0 = unstack(long)
wide1 = unstack(long, :variable, :value)
wide2 = unstack(long, :id, :variable, :value)
wide3 = unstack(long, [:id, :a], :variable, :value)
wide4 = unstack(long, :id, :variable, :value, renamecols=x->Symbol(:_, x))
```
Note that there are some differences between the widened results above.
"""
function unstack(df::AbstractDataFrame, rowkey::ColumnIndex, colkey::ColumnIndex,
                 value::ColumnIndex; renamecols::Function=identity)
    refkeycol = df[!, rowkey]
    keycol = df[!, colkey]
    valuecol = df[!, value]
    return _unstack(df, index(df)[rowkey], index(df)[colkey],
                    keycol, valuecol, refkeycol, renamecols)
end

function unstack(df::AbstractDataFrame, rowkeys, colkey::ColumnIndex,
                 value::ColumnIndex; renamecols::Function=identity)
    rowkey_ints = index(df)[rowkeys]
    @assert rowkey_ints isa AbstractVector{Int}
    length(rowkey_ints) == 0 && throw(ArgumentError("No key column found"))
    length(rowkey_ints) == 1 && return unstack(df, rowkey_ints[1], colkey, value,
                                               renamecols=renamecols)
    g = groupby(df, rowkey_ints, sort=true)
    keycol = df[!, colkey]
    valuecol = df[!, value]
    return _unstack(df, rowkey_ints, index(df)[colkey], keycol, valuecol, g, renamecols)
end

function unstack(df::AbstractDataFrame, colkey::ColumnIndex, value::ColumnIndex;
                 renamecols::Function=identity)
    colkey_int = index(df)[colkey]
    value_int = index(df)[value]
    return unstack(df, Not(colkey_int, value_int), colkey_int, value_int,
            renamecols=renamecols)
end

unstack(df::AbstractDataFrame; renamecols::Function=identity) =
    unstack(df, :variable, :value, renamecols=renamecols)

"""
    StackedVector <: AbstractVector

An `AbstractVector` that is a linear, concatenated view into
another set of AbstractVectors

NOTE: Not exported.

# Constructor
```julia
StackedVector(d::AbstractVector)
```

# Arguments
- `d...` : one or more AbstractVectors

# Examples
```julia
StackedVector(Any[[1,2], [9,10], [11,12]])  # [1,2,9,10,11,12]
```
"""
struct StackedVector{T} <: AbstractVector{T}
    components::Vector{Any}
end

StackedVector(d::AbstractVector) =
    StackedVector{promote_type(map(eltype, d)...)}(d)

function Base.getindex(v::StackedVector{T}, i::Int)::T where T
    lengths = [length(x)::Int for x in v.components]
    cumlengths = [0; cumsum(lengths)]
    j = searchsortedlast(cumlengths .+ 1, i)
    if j > length(cumlengths)
        error("indexing bounds error")
    end
    k = i - cumlengths[j]
    if k < 1 || k > length(v.components[j])
        error("indexing bounds error")
    end
    return v.components[j][k]
end

Base.IndexStyle(::Type{StackedVector}) = Base.IndexLinear()
Base.size(v::StackedVector) = (length(v),)
Base.length(v::StackedVector) = sum(map(length, v.components))
Base.eltype(v::Type{StackedVector{T}}) where {T} = T
Base.similar(v::StackedVector, T::Type, dims::Union{Integer, AbstractUnitRange}...) =
    similar(v.components[1], T, dims...)

"""
    RepeatedVector{T} <: AbstractVector{T}

An AbstractVector that is a view into another AbstractVector with
repeated elements

NOTE: Not exported.

# Constructor
```julia
RepeatedVector(parent::AbstractVector, inner::Int, outer::Int)
```

# Arguments
- `parent` : the AbstractVector that's repeated
- `inner` : the numer of times each element is repeated
- `outer` : the numer of times the whole vector is repeated after
  expanded by `inner`

`inner` and `outer` have the same meaning as similarly named arguments
to `repeat`.

# Examples
```julia
RepeatedVector([1,2], 3, 1)   # [1,1,1,2,2,2]
RepeatedVector([1,2], 1, 3)   # [1,2,1,2,1,2]
RepeatedVector([1,2], 2, 2)   # [1,2,1,2,1,2,1,2]
```
"""
struct RepeatedVector{T} <: AbstractVector{T}
    parent::AbstractVector{T}
    inner::Int
    outer::Int
end

Base.parent(v::RepeatedVector) = v.parent
DataAPI.levels(v::RepeatedVector) = levels(parent(v))

function Base.getindex(v::RepeatedVector, i::Int)
    N = length(parent(v))
    idx = Base.fld1(mod1(i,v.inner*N),v.inner)
    parent(v)[idx]
end

Base.IndexStyle(::Type{<:RepeatedVector}) = Base.IndexLinear()
Base.size(v::RepeatedVector) = (length(v),)
Base.length(v::RepeatedVector) = v.inner * v.outer * length(parent(v))
Base.eltype(v::Type{RepeatedVector{T}}) where {T} = T
Base.reverse(v::RepeatedVector) = RepeatedVector(reverse(parent(v)), v.inner, v.outer)
Base.similar(v::RepeatedVector, T::Type, dims::Dims) = similar(parent(v), T, dims)
Base.unique(v::RepeatedVector) = unique(parent(v))
